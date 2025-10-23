import os
import sys
import subprocess
import struct


def _get_slangc_path():
  vulkan_sdk = os.getenv("VULKAN_SDK")
  if vulkan_sdk:
    return os.path.join(vulkan_sdk, "bin", "slangc")
  return "slangc"


if __name__ == "__main__":
  argv = sys.argv.copy()
  argv[0] = _get_slangc_path()

  for i in range(len(argv)):
    if argv[i] == "-o=":
      path = argv[i][3:]
      break
    elif argv[i] == "-o":
      path = argv[i+1]
      break


  os.makedirs(os.path.dirname(path), exist_ok=True)
  proc = subprocess.run(argv, check=True, env=os.environ.copy(), capture_output=True)
  stdout, stderr = proc.stdout, proc.stderr

  if not os.path.exists(path):
    raise RuntimeError("\n".join([f"Output {path} not found", f"stdout: {stdout.decode() if stdout else 'None'}", f"stderr: {stderr.decode() if stderr else 'None'}"]))

  with open(path, "rb") as f:
    spv = f.read()

  data = struct.unpack(f"<{len(spv) // 4}I", spv)

  filename = os.path.splitext(os.path.basename(path))[0]
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
