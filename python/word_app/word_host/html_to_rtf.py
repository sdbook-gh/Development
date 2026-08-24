"""Convert a subset of HTML (Word-paste style) to RTF without Word/LibreOffice.

Focused on elements Word produces when copying: paragraphs, headings,
bold/italic/underline/strike, font size/family/color/highlight, lists,
tables, and inline images (data URIs). Pure stdlib, no COM.
"""

from __future__ import annotations

import base64
import re
from html.parser import HTMLParser


class _El:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag, attrs):
        self.tag = tag
        self.attrs = {k: (v or "") for k, v in attrs}
        self.children = []


_VOID = {"br", "img", "hr", "meta", "link", "input", "col", "area", "base", "wbr"}


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _El("root", [])
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        el = _El(tag, attrs)
        self.stack[-1].children.append(el)
        if tag not in _VOID:
            self.stack.append(el)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_El(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if data:
            self.stack[-1].children.append(("text", data))


_NAMED = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
    "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "silver": (192, 192, 192), "maroon": (128, 0, 0),
    "olive": (128, 128, 0), "navy": (0, 0, 128), "purple": (128, 0, 128),
    "teal": (0, 128, 128), "lime": (0, 255, 0), "aqua": (0, 255, 255),
    "fuchsia": (255, 0, 255), "orange": (255, 165, 0),
}

_HSZ = {"h1": 24.0, "h2": 18.0, "h3": 14.0, "h4": 12.0, "h5": 11.0, "h6": 10.0}
_FONTSIZE_HTML = {1: 7.5, 2: 10.0, 3: 12.0, 4: 13.5, 5: 18.0, 6: 24.0, 7: 36.0}
_BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5",
               "h6", "ul", "ol", "li", "table", "tr", "td", "th", "hr",
               "blockquote"}
def _pcolor(v):
    v = v.strip().lower()
    if not v:
        return None
    if v in _NAMED:
        return _NAMED[v]
    m = re.match(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", v)
    if not m:
        m = re.match(r"^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", v)
        if m:
            return (int(m[1]), int(m[2]), int(m[3]))
        return None
    h = m[1]
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _ppt(v):
    v = v.strip().lower()
    if not v:
        return None
    m = re.match(r"^([\d.]+)\s*(pt|px|em|rem|%)?$", v)
    if not m:
        return None
    n = float(m[1])
    u = m[2] or "pt"
    if u == "px":
        return n * 0.75
    if u in ("em", "rem"):
        return n * 12.0
    if u == "%":
        return None
    return n


def _pstyle(s):
    out = {}
    for d in s.split(";"):
        if ":" not in d:
            continue
        k, v = d.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if not k:
            continue
        if k == "font-size":
            p = _ppt(v)
            if p:
                out["size_pt"] = p
        elif k == "color":
            c = _pcolor(v)
            if c:
                out["color"] = c
        elif k == "background-color":
            c = _pcolor(v)
            if c:
                out["highlight"] = c
        elif k == "font-family":
            fam = v.split(",")[0].strip().strip("'\"").lower()
            if fam:
                out["font"] = fam
        elif k == "font-weight":
            if v.lower() in ("bold", "bolder", "700", "800", "900"):
                out["bold"] = True
        elif k == "font-style":
            if v.lower() in ("italic", "oblique"):
                out["italic"] = True
        elif k == "text-decoration":
            dd = v.lower()
            if "underline" in dd:
                out["underline"] = True
            if "line-through" in dd:
                out["strike"] = True
        elif k == "text-align":
            if v.lower() in ("left", "center", "right", "justify"):
                out["align"] = v.lower()
        elif k == "vertical-align":
            if v.lower() in ("top", "middle", "bottom"):
                out["valign"] = v.lower()
    return out


def _style_for(el, parent):
    s = dict(parent)
    t = el.tag
    if t in ("b", "strong"):
        s["bold"] = True
    elif t in ("i", "em"):
        s["italic"] = True
    elif t == "u":
        s["underline"] = True
    elif t in ("s", "strike", "del"):
        s["strike"] = True
    elif t == "sub":
        s["sub"] = True
    elif t == "sup":
        s["sup"] = True
    elif t == "mark":
        s["highlight"] = (255, 255, 0)
    elif t in ("code", "pre"):
        s["font"] = "consolas"
    elif t in _HSZ:
        s["size_pt"] = _HSZ[t]
        s["bold"] = True
    elif t == "th":
        s["bold"] = True
        s["align"] = s.get("align", "center")
    if "style" in el.attrs:
        s.update(_pstyle(el.attrs["style"]))
    if "color" in el.attrs:
        c = _pcolor(el.attrs["color"])
        if c:
            s["color"] = c
    if "bgcolor" in el.attrs:
        c = _pcolor(el.attrs["bgcolor"])
        if c:
            s["highlight"] = c
    if el.attrs.get("align") in ("left", "center", "right", "justify"):
        s["align"] = el.attrs["align"]
    if el.attrs.get("valign") in ("top", "middle", "bottom"):
        s["valign"] = el.attrs["valign"]
    if t == "font":
        if "face" in el.attrs:
            fam = el.attrs["face"].split(",")[0].strip().strip("'\"").lower()
            if fam:
                s["font"] = fam
        if "size" in el.attrs:
            try:
                s["size_pt"] = _FONTSIZE_HTML.get(int(el.attrs["size"]), 12.0)
            except ValueError:
                pass
    return s
class _R:
    def __init__(self):
        self.fonts = ["Microsoft YaHei"]
        self.fidx = {"microsoft yahei": 0}
        self.colors = [(0, 0, 0)]
        self.cidx = {(0, 0, 0): 0}

    def _font(self, n):
        k = n.lower()
        if k in self.fidx:
            return self.fidx[k]
        i = len(self.fonts)
        self.fonts.append(n)
        self.fidx[k] = i
        return i

    def _color(self, c):
        if c in self.cidx:
            return self.cidx[c]
        i = len(self.colors)
        self.colors.append(c)
        self.cidx[c] = i
        return i

    @staticmethod
    def _esc(text):
        out = []
        for ch in text:
            o = ord(ch)
            if ch == "\\":
                out.append("\\\\")
            elif ch == "{":
                out.append("\\{")
            elif ch == "}":
                out.append("\\}")
            elif ch == "\n":
                out.append("\\line ")
            elif ch == "\r":
                continue
            elif ch == "\t":
                out.append("\\tab ")
            elif o < 128:
                out.append(ch)
            else:
                code = o - 65536 if o > 32767 else o
                out.append(f"\\u{code}?")
        return "".join(out)

    def _run(self, text, style):
        if not text:
            return ""
        p = []
        if style.get("bold"):
            p.append("\\b")
        if style.get("italic"):
            p.append("\\i")
        if style.get("underline"):
            p.append("\\ul")
        if style.get("strike"):
            p.append("\\strike")
        if style.get("super"):
            p.append("\\super")
        elif style.get("sub"):
            p.append("\\sub")
        if "size_pt" in style:
            p.append(f"\\fs{int(round(style['size_pt'] * 2))}")
        if "font" in style:
            p.append(f"\\f{self._font(style['font'])}")
        if "color" in style:
            p.append(f"\\cf{self._color(style['color'])}")
        if "highlight" in style:
            p.append(f"\\highlight{self._color(style['highlight'])}")
        return "{" + "".join(p) + " " + self._esc(text) + "}"
    def _inline(self, node, style, out):
        if isinstance(node, tuple) and node[0] == "text":
            r = self._run(node[1], style)
            if r:
                out.append(r)
            return
        if not isinstance(node, _El):
            return
        t = node.tag
        if t == "br":
            out.append("\\line ")
            return
        if t == "img":
            self._img(node, out)
            return
        ns = _style_for(node, style)
        if t in ("p", "div", "section", "article", "blockquote") or t in _HSZ:
            sub = []
            self._block(node, ns, sub)
            out.extend(sub)
            return
        for c in node.children:
            self._inline(c, ns, out)

    def _img(self, el, out):
        src = el.attrs.get("src", "")
        m = re.match(r"^data:image/(\w+);base64,(.+)$", src, re.S)
        if not m:
            return
        kind = m[1].lower()
        try:
            data = base64.b64decode(m[2])
        except Exception:
            return
        blip = "\\pngblip" if kind in ("png", "apng") else "\\jpegblip"
        hexs = data.hex()
        w = h = ""
        try:
            if "width" in el.attrs:
                wpx = float(re.sub(r"[^0-9.]", "", el.attrs["width"]) or 0)
                if wpx:
                    w = f"\\picw{int(wpx)}\\picwgoal{int(wpx * 15)}"
            if "height" in el.attrs:
                hpx = float(re.sub(r"[^0-9.]", "", el.attrs["height"]) or 0)
                if hpx:
                    h = f"\\pich{int(hpx)}\\pichgoal{int(hpx * 15)}"
        except Exception:
            w = h = ""
        out.append("{\\pict" + blip + w + h + " " + hexs + "}")

    def _align(self, style):
        a = style.get("align")
        if a == "center":
            return "\\qc"
        if a == "right":
            return "\\qr"
        if a == "justify":
            return "\\qj"
        return "\\ql"
    def _block(self, el, style, out):
        t = el.tag
        if t in ("ul", "ol"):
            self._list(el, style, out, ordered=(t == "ol"))
            return
        if t == "table":
            self._table(el, style, out)
            return
        if t in ("tr", "td", "th"):
            for c in el.children:
                if isinstance(c, _El) and c.tag in _BLOCK_TAGS:
                    self._block(c, _style_for(c, style), out)
                else:
                    runs = []
                    self._inline(c, style, runs)
                    if runs:
                        out.append("{\\pard" + self._align(style) + " " + "".join(runs) + "\\par}")
            return
        if t == "hr":
            out.append("{\\pard\\brdrb\\brdrs\\brdrw15\\brsp20 \\par}")
            return
        runs = []
        for c in el.children:
            if isinstance(c, _El) and c.tag in _BLOCK_TAGS and c.tag != "li":
                if runs:
                    out.append("{" + "\\pard" + self._align(style) + " " + "".join(runs) + "\\par}")
                    runs = []
                self._block(c, _style_for(c, style), out)
            else:
                self._inline(c, style, runs)
        if runs or (t in ("p", "div") and not el.children):
            out.append("{" + "\\pard" + self._align(style) + " " + "".join(runs) + "\\par}")

    def _list(self, el, style, out, ordered):
        i = 0
        for c in el.children:
            if isinstance(c, _El) and c.tag in ("ul", "ol"):
                self._list(c, style, out, ordered=(c.tag == "ol"))
                continue
            if not isinstance(c, _El) or c.tag != "li":
                continue
            i += 1
            ns = _style_for(c, style)
            runs = []
            for cc in c.children:
                if isinstance(cc, _El) and cc.tag in ("ul", "ol"):
                    if runs:
                        prefix = f"{i}. " if ordered else "\\u8226?  "
                        out.append("{\\pard\\fi-360\\li360 " + prefix + "".join(runs) + "\\par}")
                        runs = []
                    self._list(cc, ns, out, ordered=(cc.tag == "ol"))
                else:
                    self._inline(cc, ns, runs)
            prefix = f"{i}. " if ordered else "\\u8226?  "
            out.append("{\\pard\\fi-360\\li360 " + prefix + "".join(runs) + "\\par}")
    def _table(self, el, style, out):
        rows = [c for c in el.children if isinstance(c, _El) and c.tag == "tr"]
        if not rows:
            for c in el.children:
                if isinstance(c, _El) and c.tag in ("thead", "tbody", "tfoot"):
                    rows.extend(x for x in c.children if isinstance(x, _El) and x.tag == "tr")
            if not rows:
                return
        ncol = 0
        for r in rows:
            ncol = max(ncol, sum(1 for c in r.children if isinstance(c, _El) and c.tag in ("td", "th")))
        if ncol == 0:
            return
        usable = 9360
        cw = usable // ncol
        for r in rows:
            cells = [c for c in r.children if isinstance(c, _El) and c.tag in ("td", "th")]
            row_parts = ["\\trowd\\trgaph60\\trleft0\\trrh0"]
            x = 0
            for ci in range(ncol):
                x += cw
                row_parts.append(f"\\cellx{x}")
            for ci in range(ncol):
                cell = cells[ci] if ci < len(cells) else _El("td", [])
                cs = _style_for(cell, style)
                va = cs.get("valign")
                vctl = "\\clvertalt" if va == "top" else ("\\clvertalc" if va == "middle" else "\\clvertalb")
                row_parts.append(vctl)
                runs = []
                for cc in cell.children:
                    if isinstance(cc, _El) and cc.tag in _BLOCK_TAGS and cc.tag not in ("td", "th", "tr"):
                        sub = []
                        self._block(cc, _style_for(cc, cs), sub)
                        runs.append("".join(sub))
                    else:
                        self._inline(cc, cs, runs)
                row_parts.append("\\pard\\intbl" + self._align(cs) + " " + "".join(runs) + "\\cell")
            row_parts.append("\\row")
            out.append("{" + "".join(row_parts) + "}")

    def render(self, root):
        body = []
        for c in root.children:
            if isinstance(c, _El) and c.tag in _BLOCK_TAGS:
                self._block(c, _style_for(c, {}), body)
            elif isinstance(c, _El) and c.tag in ("html", "body", "head"):
                self.render(c)
            else:
                runs = []
                self._inline(c, {}, runs)
                if runs:
                    body.append("{\\pard\\ql " + "".join(runs) + "\\par}")
        fonttbl = []
        for i, name in enumerate(self.fonts):
            fonttbl.append("{\\f" + str(i) + " " + name + ";}")
        colortbl = [";"]
        for (r, g, b) in self.colors[1:]:
            colortbl.append("\\red" + str(r) + "\\green" + str(g) + "\\blue" + str(b) + ";")
        header = (
            "{\\rtf1\\ansi\\ansicpg936\\deff0\n"
            "{\\fonttbl" + "".join(fonttbl) + "}\n"
            "{\\colortbl" + "".join(colortbl) + "}\n"
            "\\paperw12240\\paperh15840\\margl1440\\margr1440\\margt1440\\margb1440\n"
            "\\f0\\fs22\\fi0\\li0\n"
        )
        return header + "".join(body) + "}"


def html_to_rtf(html: str) -> str:
    tb = _TreeBuilder()
    tb.feed(html or "")
    tb.close()
    return _R().render(tb.root)
# __END__
