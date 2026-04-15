import os
import pickle
import re

from tqdm import tqdm


def _normalize_tuh_join(base_dir, rel_path):
    rel = rel_path.replace("\\", "/").lstrip("/")
    return os.path.normpath(os.path.join(base_dir, rel))


def read_headers(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    elif file_path.endswith(".tar.gz"):
        import tarfile

        with tarfile.open(file_path, "r:gz") as tar:
            filenames = tar.getnames()
            file_name = [name for name in filenames if name.endswith(".txt")][0]
            print(f"Extracting {file_name} from the archive...")
            f = tar.extractfile(file_name)
            content = f.read().decode("utf-8")
    else:
        raise ValueError("Unsupported file format. Please provide a .txt or .tar.gz file.")

    print(f"File read successfully. Total size: {len(content) / 1024 / 1024:.1f} MB")
    return content


def split_entries(content, split_regex=r"\s*\d+:\s+/data/isip/www/isip/", include_regex=False):
    if include_regex:
        split_regex = f"({split_regex})"
        entries = re.split(split_regex, content, flags=re.MULTILINE)[1:]
        entries = [f"{entries[i]}{entries[i + 1]}" for i in range(0, len(entries), 2)]
    else:
        entries = re.split(split_regex, content, flags=re.MULTILINE)[1:]
    print(f"Found {len(entries)} entries in the file.")
    return entries


def _parser_line_by_line(content, splitby="=", key_lambda=lambda k: k, values_lambda=lambda vlist: vlist):
    parsed = {}
    for line in content.splitlines():
        if splitby in line:
            key, values = map(str.strip, line.split(splitby, 1))
            parsed[key_lambda(key)] = values_lambda(values.strip("[]"))
    return parsed


def parse_entry(entry_text):
    entry_dict = {"record": {}}
    entry_number = re.search(r"\d+", entry_text)
    if entry_number:
        entry_dict["record"]["ID"] = entry_number.group(0)

    match = re.search(r"(/[^\n]+)", entry_text)
    if match:
        full_path = match.group(1)
        relative_path = full_path.split("/data/isip/www/isip/", 1)[-1]
        entry_dict["record"]["relative_path"] = "\\isip.piconepress.com\\" + relative_path

    block_regex = r"\n\tBlock (\d+):\s*(.+?)\n\t"
    block_regex = f"({block_regex})"
    blocks = re.split(block_regex, entry_text, flags=re.MULTILINE)
    for i in range(1, len(blocks), 4):
        block_number = blocks[i + 1]
        block_desc = blocks[i + 2]
        block_content = blocks[i + 3]
        entry_dict[block_desc] = {"block_number": int(block_number)}

        if block_number == "1":
            version = re.search(r"\[.*?\]", block_content)
            parsed_block = {"version": version.group(0).strip(" []") if version else "Parse error"}
        elif block_number in {"2", "3", "4"}:
            parsed_block = _parser_line_by_line(block_content)
        elif block_number == "5":
            parsed_block = _parser_line_by_line(
                block_content,
                key_lambda=lambda k: k.split(" ")[0],
                values_lambda=lambda vlist: [v.strip("[]") for v in re.findall(r"\[.*?\]", vlist)],
            )
        elif block_number == "6":
            parsed_equal = _parser_line_by_line(block_content, splitby="=")
            parsed_colon = _parser_line_by_line(block_content, splitby=":")
            parsed_equal["per channel sample frequencies"] = [v for v in parsed_colon.values() if v]
            parsed_block = parsed_equal
        else:
            raise NotImplementedError(f"Block {block_number} not implemented.")
        entry_dict[block_desc].update(parsed_block)
    return entry_dict


def custom_format_entry(entry_dict):
    path_ = entry_dict["record"]["relative_path"].replace("/nedc/data/eeg/", "/nedc/data/tuh_eeg/")
    path_ = path_.replace("v2.0.0", "v2.0.1")
    path_ = re.sub(r"/s(\d{3})_(\d{4})_(\d{2})_(\d{2})/", r"/s\1_\2/", path_)
    path_ = path_.replace("\\", "/").replace("//", "/")
    entry_dict["record"]["relative_path"] = path_[1:] if path_.startswith("/") else path_
    entry_dict["record"]["ID"] = int(entry_dict["record"]["ID"])
    return entry_dict


def parse_headers_to_pickle(
    header_path,
    output_path,
    reparse=True,
):
    output_path = os.path.abspath(os.path.normpath(output_path))
    header_abs_path = os.path.abspath(os.path.normpath(header_path))
    if not os.path.exists(header_abs_path):
        raise FileNotFoundError(f"Local headers file not found: {header_abs_path}")

    header_report = {
        "absolute_path": header_abs_path,
        "exists": True,
    }
    if not reparse and os.path.exists(output_path):
        with open(output_path, "rb") as f:
            parsed_header = pickle.load(f)
        return parsed_header, output_path, header_report

    content = read_headers(header_abs_path)
    entries = split_entries(content, include_regex=True)
    parsed = [custom_format_entry(parse_entry(entry)) for entry in tqdm(entries, desc="Parsing and formatting entries")]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(parsed, f)

    return parsed, output_path, header_report
