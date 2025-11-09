# BoxScope A91 helper

This repository now contains a standalone Python utility that replicates the
behaviour of the one-shot shell script shared in the task description.  The
script no longer depends on `pip`-installed packages at runtime – it communicates
with the OpenAI API using only the Python standard library, which makes it work
in restricted environments where outbound package downloads are blocked.

## Usage

```bash
export OPENAI_API_KEY=sk-...
python scripts/boxscope_a91.py "https://furusetalle9.oslo.no/inventory/photos/A91/" "~/wynik/A91"
```

The script accepts the same environment variables as the original tool.  You can
override the ignore regex or the maximum number of objects by exporting
`IGNORE_REGEX`, `MAX`, etc.  The generated JSON/CSV/HTML files are written to the
output directory (created if necessary).

Local images are supported as well: provide either a direct path to an image file
or a directory containing supported formats, and the script will upload the files
as data URLs.

## Quick smoke test

You can confirm that everything is wired up correctly without burning through the
entire catalogue.  The example below limits the run to the very first image,
disables additional rotations/tiling, and writes the artefacts to a temporary
directory so that you can inspect the output afterwards.

```bash
export OPENAI_API_KEY=sk-...
LIMIT=1 ROTATIONS=0 TILES=1x1 python scripts/boxscope_a91.py \
  "https://furusetalle9.oslo.no/inventory/photos/A91/" \
  "$(mktemp -d)/boxscope-demo"
```

When the command finishes you should see a line similar to `Results written to
/tmp/boxscope-demo...`.  Inspect the generated `_index.json` or open the HTML
report in a browser to confirm that the API calls succeeded.  Increase `LIMIT`
again when you're ready for a full catalogue.

## Offline self-test

If you simply want to verify that the Python pipeline works end-to-end (without
contacting the OpenAI API) run the built-in self-test mode.  It feeds an
embedded 1×1 PNG through the same aggregation code path and produces artefacts in
the chosen directory.

```bash
python scripts/boxscope_a91.py --self-test "$(mktemp -d)/boxscope-self-test"
```

The resulting JSON/CSV/HTML files contain a single synthetic entry labelled
"test cube".  Seeing those files confirms that local I/O, aggregation and HTML
rendering all work even without internet access or an API key.
