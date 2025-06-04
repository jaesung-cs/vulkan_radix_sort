import os
import sys
import subprocess
import struct

if __name__ == "__main__":
  argv = sys.argv.copy()
  argv[0] = "slangc"
  subprocess.run(argv, check=True)

  for i in range(len(argv)):
    if argv[i] == "-o=":
      path = argv[i][3:]
      break
    elif argv[i] == "-o":
      path = argv[i+1]
      break

  filename = os.path.splitext(os.path.basename(path))[0]

  with open(path, "rb") as f:
    spv = f.read()

  data = struct.unpack(f"<{len(spv) // 4}I", spv)

  code = "#pragma once\n"
  code += "#include <cstdint>\n"
  code += f"const uint32_t {filename}[] = {{\n"
  for i in range(0, len(data), 8):
    words = [f"0x{x:08x}" for x in data[i:i+8]]
    line = "  " + ",".join(words) + ",\n"
    code += line
  code += "};\n"

  with open(path, "w") as f:
    f.write(code)
