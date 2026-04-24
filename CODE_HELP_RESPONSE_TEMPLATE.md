# Code Help Response Template

When discussing how to implement a function, use the following 3 dimensions:

## 1) Data Flow
- Input: source, type, shape/format
- Output: type, shape/format
- Conversion rules: how input is transformed into output
- Must include realistic example data (not only abstract types)
- Prefer showing a mini before/after example for one function call

## 2) Role in Pipeline
- Why this function exists
- Which previous step provides its input
- Which next step consumes its output
- What breaks if this function is missing

## 3) Required Libraries and APIs
- Which libraries/modules are needed
- Which key functions are used
- Minimal usage examples and common pitfalls
- Prefer runnable code snippets whenever possible
- Show at least one direct snippet for each critical API used

---

## Example: `split_wav_and_label(wav_files, label_dict)`

### 1) Data Flow
- Input:
  - `wav_files`: `List[Path]`, usually from `list_wav_files(...)`
  - `label_dict`: `Dict[str, List[int]]`, usually from `read_label_from_file(...)`
- Realistic input example:
```python
wav_files = [
    Path(".../wavs/dev/1031-133220-0062.wav"),
    Path(".../wavs/dev/9999-000000-0001.wav"),
]
label_dict = {
    "1031-133220-0062": [0, 0, 1, 1, 1, 0],
    "1051-133881-0021": [0, 1, 1, 0],
}
```
- Output:
  - `List[Tuple[Path, List[int]]]`
  - Each tuple is `(wav_path, frame_labels)` for one matched utterance
- Realistic output example:
```python
pairs = [
    (Path(".../wavs/dev/1031-133220-0062.wav"), [0, 0, 1, 1, 1, 0])
]
```
- Core transform:
  - Use `utt_id = wav_path.stem`
  - Match by `utt_id in label_dict`
  - Keep only matched pairs

### 2) Role in Pipeline
- Why needed:
  - It aligns audio samples with supervision labels.
- Upstream:
  - wav paths from disk scan
  - labels from label text parsing
- Downstream:
  - training/dev loop iterates over `(wav, label)` pairs
- If missing:
  - model/training code cannot know which label belongs to which audio
  - easy to produce label-audio mismatch bugs

### 3) Required Libraries and APIs
- `pathlib.Path`
  - `wav_path.stem` gets utterance id without `.wav`
- Direct snippet:
```python
utt_id = wav_path.stem
```
- Python container ops
  - `if key in dict`
  - `list.append(...)`
- Direct snippets:
```python
if utt_id in label_dict:
    pairs.append((wav_path, label_dict[utt_id]))
```
- Optional diagnostics
  - `set(...)` for unmatched id statistics
- Direct snippet:
```python
matched_ids = {p[0].stem for p in pairs}
unused_label_ids = set(label_dict) - matched_ids
```
