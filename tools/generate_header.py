import os
import re
import sys

SHADER_TAG = re.compile(r'^// @SHADER_DATA:(\w+)@\n?$')
VERSION_TAG = re.compile(r'@VERSION_MAJOR@|@VERSION_MINOR@|@VERSION_PATCH@')

if __name__ == "__main__":
    template_path, output_path, generated_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    version_major, version_minor, version_patch = sys.argv[4], sys.argv[5], sys.argv[6]

    version_map = {
        '@VERSION_MAJOR@': version_major,
        '@VERSION_MINOR@': version_minor,
        '@VERSION_PATCH@': version_patch,
    }

    with open(template_path, 'r') as f:
        template_lines = f.readlines()

    result = []
    for line in template_lines:
        m = SHADER_TAG.match(line)
        if m:
            name = m.group(1)
            header_path = os.path.join(generated_dir, f"{name}.h")
            with open(header_path, 'r') as f:
                shader_lines = f.readlines()
            # Strip preamble (#pragma once, #include lines)
            body = [l for l in shader_lines if not l.startswith('#pragma') and not l.startswith('#include')]
            result.extend(body)
        else:
            for placeholder, value in version_map.items():
                line = line.replace(placeholder, value)
            result.append(line)

    with open(output_path, 'w') as f:
        f.writelines(result)
