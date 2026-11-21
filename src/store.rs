//! The "neocortex" boundary: durable payload storage behind a trait. Engram
//! stores sparse codes + graph structure and carries a small **handle** on each
//! episode; turning that handle back into content is the store's job (settled
//! decision 11).
//!
//! The handle (`PayloadRef`) is opaque to engram — a row id, a UUID, an index,
//! whatever your durable store keys on. Because it lives inline in every episode
//! and in the snapshot, it must be `Copy` and serialize to a fixed byte width.
//! Big or variable-size content goes *behind* the store, keyed by the handle.

use alloc::vec::Vec;
use core::marker::PhantomData;

/// A fixed-size, copyable handle engram stores on each episode and hands back on
/// recall. Implement it for your own id type (a `Uuid` newtype is ~4 lines);
/// impls for the common integer widths and `[u8; 16]` ship below.
/// `Ord` because the graph keeps a payload → slot index (relational state
/// export/import is keyed by payload — the only cross-device-stable identity).
pub trait PayloadRef: Copy + Eq + Ord {
    /// Serialized width in bytes. Fixed on purpose — a variable-size ref would
    /// bloat the hot-path episode; put variable content behind the store.
    const SIZE: usize;
    fn write_bytes(&self, out: &mut Vec<u8>);
    /// `bytes.len() >= SIZE` is guaranteed by the caller.
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

macro_rules! impl_payload_ref_int {
    ($($t:ty),*) => {$(
        impl PayloadRef for $t {
            const SIZE: usize = core::mem::size_of::<$t>();
            #[inline]
            fn write_bytes(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            #[inline]
            fn from_bytes(bytes: &[u8]) -> Option<Self> {
                let arr: [u8; core::mem::size_of::<$t>()] =
                    bytes.get(..Self::SIZE)?.try_into().ok()?;
                Some(<$t>::from_le_bytes(arr))
            }
        }
    )*};
}
impl_payload_ref_int!(u32, i32, u64, i64, u128, i128);

/// A raw 16-byte UUID (e.g. `uuid::Uuid::as_bytes`).
impl PayloadRef for [u8; 16] {
    const SIZE: usize = 16;
    #[inline]
    fn write_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
    #[inline]
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bytes.get(..16)?.try_into().ok()
    }
}

/// The durable store — the neocortex. Engram never inspects `Payload`; it only
/// moves handles around.
///
/// # A SQLite backend (interface only — engram pulls in no SQL dependency)
///
/// ```ignore
/// struct SqliteStore { conn: rusqlite::Connection } // in YOUR crate
///
/// impl PayloadStore for SqliteStore {
///     type Ref = i64;        // the row's INTEGER PRIMARY KEY (rowid)
///     type Payload = String; // or Vec<u8>, or a serialized record
///
///     fn put(&mut self, body: String) -> i64 {
///         self.conn
///             .execute("INSERT INTO engram_payload(body) VALUES (?1)", (&body,))
///             .unwrap();
///         self.conn.last_insert_rowid()          // Ref == the real rowid: no second id
///     }
///     fn get(&self, id: i64) -> Option<String> {
///         self.conn
///             .query_row("SELECT body FROM engram_payload WHERE id=?1", (id,), |r| r.get(0))
///             .ok()
///     }
///     fn remove(&mut self, id: i64) {
///         // Safe to DELETE only because this table is OWNED by engram.
///         self.conn.execute("DELETE FROM engram_payload WHERE id=?1", (id,)).ok();
///     }
/// }
/// ```
///
/// If instead you key on rows your *app* owns (chat messages, etc.), use
/// [`IdentityStore`]: engram carries your id and hands it back, and `remove`
/// stays a no-op so forgetting a trace never deletes your source data.
pub trait PayloadStore {
    /// The handle stored on the episode and returned by recall.
    type Ref: PayloadRef;
    /// What the caller stores and gets back.
    type Payload;

    fn put(&mut self, payload: Self::Payload) -> Self::Ref;

    /// Owned return — a database backend builds a fresh value from a row; it
    /// cannot lend a borrow that outlives the call. (This is why the old
    /// `-> Option<&str>` shape was in-memory-only.)
    fn get(&self, r: Self::Ref) -> Option<Self::Payload>;

    /// Called when engram **prunes** the episode holding `r` (a forgotten
    /// trace). Reclaim storage you OWN here. If `r` points at data owned
    /// elsewhere (your app's rows), leave this as the default no-op — otherwise
    /// forgetting a trace would delete your source data.
    fn remove(&mut self, _r: Self::Ref) {}
}

/// In-memory reference store: engram assigns a `u64` index as the handle and
/// keeps the payload in a slab. `remove` frees the slot (and it is reused), so
/// forgetting actually reclaims — no unbounded growth as the graph prunes.
pub struct InMemoryStore<P = alloc::string::String> {
    slots: Vec<Option<P>>,
    free: Vec<u64>,
}

impl<P> InMemoryStore<P> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Number of live payloads (excludes freed slots).
    pub fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<P> Default for InMemoryStore<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Clone> PayloadStore for InMemoryStore<P> {
    type Ref = u64;
    type Payload = P;

    fn put(&mut self, payload: P) -> u64 {
        match self.free.pop() {
            Some(i) => {
                self.slots[i as usize] = Some(payload);
                i
            }
            None => {
                self.slots.push(Some(payload));
                (self.slots.len() - 1) as u64
            }
        }
    }

    fn get(&self, r: u64) -> Option<P> {
        self.slots.get(r as usize).and_then(|s| s.clone())
    }

    fn remove(&mut self, r: u64) {
        if let Some(slot) = self.slots.get_mut(r as usize) {
            if slot.take().is_some() {
                self.free.push(r);
            }
        }
    }
}

/// Pass-through store for callers that already own a stable id (a UUID, a row
/// key). The handle *is* your id: `put` and `get` are the identity, engram
/// stores nothing, and `remove` is a no-op — engram forgetting an episode must
/// never touch data it doesn't own. `recall` hands your ids straight back; you
/// resolve id → content in your own database.
#[derive(Clone, Copy, Debug)]
pub struct IdentityStore<R = u64>(PhantomData<R>);

impl<R> IdentityStore<R> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<R> Default for IdentityStore<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: PayloadRef> PayloadStore for IdentityStore<R> {
    type Ref = R;
    type Payload = R;

    fn put(&mut self, payload: R) -> R {
        payload
    }

    fn get(&self, r: R) -> Option<R> {
        Some(r)
    }
    // remove: default no-op — never delete caller-owned data.
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};

    #[test]
    fn payload_ref_roundtrips() {
        for v in [0u64, 1, 42, u64::MAX] {
            let mut buf = Vec::new();
            v.write_bytes(&mut buf);
            assert_eq!(buf.len(), u64::SIZE);
            assert_eq!(u64::from_bytes(&buf), Some(v));
        }
        let uuid = [7u8; 16];
        let mut buf = Vec::new();
        uuid.write_bytes(&mut buf);
        assert_eq!(<[u8; 16]>::from_bytes(&buf), Some(uuid));
    }

    #[test]
    fn in_memory_put_get_remove_reuses_slot() {
        let mut s: InMemoryStore<String> = InMemoryStore::new();
        let a = s.put("first".to_string());
        let b = s.put("second".to_string());
        assert_eq!(s.get(a).as_deref(), Some("first"));
        assert_eq!(s.len(), 2);
        s.remove(a);
        assert_eq!(s.get(a), None);
        assert_eq!(s.len(), 1);
        let c = s.put("third".to_string()); // reuses a's freed slot
        assert_eq!(c, a);
        assert_eq!(s.get(b).as_deref(), Some("second"));
    }

    #[test]
    fn identity_store_is_passthrough() {
        let mut s: IdentityStore<u128> = IdentityStore::new();
        let id = 0xDEAD_BEEF_u128;
        assert_eq!(s.put(id), id);
        assert_eq!(s.get(id), Some(id));
        s.remove(id); // no-op, must not panic
        assert_eq!(s.get(id), Some(id));
    }
}
