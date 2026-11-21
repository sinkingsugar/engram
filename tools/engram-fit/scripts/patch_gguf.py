#!/usr/bin/env python3
"""Copy a GGUF, adding tokenizer.ggml.token_type_count metadata.

Old BERT-family conversions predate llama.cpp requiring this key.
Usage: patch_gguf.py input.gguf output.gguf token_type_count
"""
import sys

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

TT_KEY = "tokenizer.ggml.token_type_count"


def field_value(field):
    """Parsed contents across gguf-py versions."""
    if hasattr(field, "contents"):
        return field.contents()
    # Fallback for older gguf-py: manual parts decoding
    types = field.types
    if types[0] == GGUFValueType.STRING:
        return bytes(field.parts[-1]).decode("utf-8")
    if types[0] == GGUFValueType.ARRAY:
        if types[1] == GGUFValueType.STRING:
            return [
                bytes(field.parts[i]).decode("utf-8") for i in field.data
            ]
        return [field.parts[i].tolist()[0] for i in field.data]
    return field.parts[-1].tolist()[0]


def main(inp, outp, tt_count):
    reader = GGUFReader(inp)
    arch_field = reader.fields["general.architecture"]
    arch = field_value(arch_field)
    print(f"arch: {arch}")

    if TT_KEY in reader.fields:
        print(f"{TT_KEY} already present — nothing to do")
        return 1

    writer = GGUFWriter(path=None, arch=arch, endianess=reader.endianess)

    for field in reader.fields.values():
        # Skip keys the writer emits itself.
        if field.name in ("general.architecture", "GGUF.version",
                          "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        val = field_value(field)
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        try:
            writer.add_key_value(field.name, val, vtype, sub_type=sub_type)
        except TypeError:
            writer.add_key_value(field.name, val, vtype)

    writer.add_key_value(TT_KEY, int(tt_count), GGUFValueType.UINT32)
    print(f"added {TT_KEY} = {tt_count}")

    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
        )

    writer.write_header_to_file(path=outp)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
