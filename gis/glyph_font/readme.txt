# https://github.com/notofonts/noto-cjk
wget https://cdn.jsdelivr.net/gh/notofonts/noto-cjk@main/Sans/Variable/TTF/Subset/NotoSansSC-VF.ttf
python3 fonts/generate_pbf.py NotoSansSC-VF.ttf fonts/data
python3 fonts/serve_glyphs.py

# font_config.json
{
  "port": 8080,
  "cache_dir": "fonts/.cache",
  "fontnik_script": "fonts/generate_range.js",
  "stacks": {
    "Noto Sans Regular": { // same font name as maplibre layer configuration
      "pbf_dir": "fonts/data",
      "ttf": "NotoSansSC-VF.ttf"
    }
  }
}
