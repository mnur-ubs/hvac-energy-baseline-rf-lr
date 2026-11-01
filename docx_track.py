"""Edit document.xml (hasil unzip .docx) dengan Track Changes tingkat kata.

Elemen non-teks (sitasi Mendeley/w:sdt, gambar, field) ditampilkan sebagai ⟦k⟧ di repr()
dan dipertahankan apa adanya selama placeholder-nya tetap ada di teks baru.
"""
import copy, difflib, re
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
q = lambda t: f"{{{W}}}{t}"
INVISIBLE = {q("bookmarkStart"), q("bookmarkEnd"), q("proofErr"), q("permStart"), q("permEnd"),
             q("commentRangeStart"), q("commentRangeEnd")}
TOKEN = re.compile(r"⟦\d+⟧|\w+|\s+|[^\w\s]")


class Tracker:
    def __init__(self, author, date, start_id=9000):
        self.author, self.date, self.next_id = author, date, start_id

    def mark(self, tag):
        el = etree.Element(q(tag)); el.set(q("id"), str(self.next_id)); self.next_id += 1
        el.set(q("author"), self.author); el.set(q("date"), self.date)
        return el

    # ------------------------------------------------------------------ atom model
    @staticmethod
    def _atoms(p):
        """List atom: ('c', char, rPr) | ('o', element) | ('i', element) (invisible)."""
        out = []
        for c in p:
            if c.tag == q("pPr"):
                continue
            if c.tag == q("r"):
                kids = [k for k in c if k.tag not in (q("rPr"), q("lastRenderedPageBreak"))]
                if kids and all(k.tag in (q("t"), q("tab")) for k in kids):
                    rpr = c.find(q("rPr"))
                    for k in kids:
                        if k.tag == q("tab"):
                            out.append(("c", "\t", rpr))
                        else:
                            out.extend(("c", ch, rpr) for ch in (k.text or ""))
                    continue
                if not kids:
                    continue
            out.append(("i" if c.tag in INVISIBLE else "o", c))
        return out

    def repr(self, p):
        s, k = [], 0
        for a in self._atoms(p):
            if a[0] == "c": s.append(a[1])
            elif a[0] == "o": s.append(f"⟦{k}⟧"); k += 1
        return "".join(s)

    def text(self, p):
        return re.sub(r"⟦\d+⟧", "", self.repr(p))

    # ------------------------------------------------------------------ run builders
    def _run(self, txt, rpr, deleted=False):
        r = etree.Element(q("r"))
        if rpr is not None:
            rp = copy.deepcopy(rpr)
            for x in rp.findall(q("rPrChange")): rp.remove(x)
            r.append(rp)
        for part in re.split(r"(\t)", txt):
            if part == "\t":
                r.append(etree.Element(q("tab")))
            elif part:
                t = etree.SubElement(r, q("delText" if deleted else "t")); t.text = part; t.set(XML_SPACE, "preserve")
        return r

    def _wrap(self, kind, runs):
        m = self.mark(kind)
        for r in runs: m.append(r)
        return m

    def _del_opaque(self, el):
        el = copy.deepcopy(el)
        if el.tag == q("r"):
            return self._wrap("del", [el])
        for r in list(el.iter(q("r"))):
            for t in r.findall(q("t")): t.tag = q("delText")
            par = r.getparent(); i = par.index(r); par.remove(r)
            par.insert(i, self._wrap("del", [r]))
        return el

    def _ins_opaque(self, el):
        el = copy.deepcopy(el)
        if el.tag == q("r"):
            return self._wrap("ins", [el])
        for r in list(el.iter(q("r"))):
            par = r.getparent(); i = par.index(r); par.remove(r); par.insert(i, self._wrap("ins", [r]))
        return el

    # ------------------------------------------------------------------ paragraph edit
    def edit(self, p, new, rpr_override=None):
        """Ganti isi paragraf p menjadi `new` (format repr) dengan ins/del tingkat kata."""
        atoms = self._atoms(p)
        vis = [a for a in atoms if a[0] != "i"]
        # anchor invisible markers ke indeks atom visible sebelumnya
        anchors, last = {}, -1
        for a in atoms:
            if a[0] == "i": anchors.setdefault(last, []).append(a[1])
            else: last += 1
        units, opq, k = [], {}, 0
        for i, a in enumerate(vis):
            if a[0] == "c": units.append((a[1], i))
            else:
                ph = f"⟦{k}⟧"; opq[ph] = a[1]; k += 1
                units.extend((ch, i) for ch in ph)
        old_s = "".join(u[0] for u in units)
        if old_s == new:
            return False
        old_t, new_t = TOKEN.findall(old_s), TOKEN.findall(new)
        assert "".join(old_t) == old_s and "".join(new_t) == new
        pos, spans = 0, []
        for t in old_t:
            idx = sorted({units[j][1] for j in range(pos, pos + len(t))}); spans.append(idx); pos += len(t)
        out = []                                   # list of (status, item) where item = atom index or text
        sm = difflib.SequenceMatcher(None, old_t, new_t, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ("equal", "delete", "replace"):
                st = "eq" if op == "equal" else "del"
                for t in range(i1, i2):
                    for ai in spans[t]: out.append((st, ai))
            if op in ("insert", "replace"):
                out.append(("ins", "".join(new_t[j1:j2])))
        # bangun ulang
        for c in [c for c in p if c.tag != q("pPr")]: p.remove(c)
        for m in anchors.get(-1, []): p.append(m)
        last_rpr = next((a[2] for a in vis if a[0] == "c"), None)
        buf, buf_key = [], None

        def flush():
            nonlocal buf, buf_key
            if not buf: return
            st, rpr = buf_key; txt = "".join(buf)
            r = self._run(txt, rpr, deleted=(st == "del"))
            p.append(self._wrap("del", [r]) if st == "del" else r)
            buf, buf_key = [], None

        emitted = set()
        for st, item in out:
            if st == "ins":
                flush()
                for piece in re.split(r"(⟦\d+⟧)", item):
                    if not piece: continue
                    if piece in opq: p.append(self._ins_opaque(opq[piece]))
                    else: p.append(self._wrap("ins", [self._run(piece, rpr_override if rpr_override is not None else last_rpr)]))
                continue
            a = vis[item]
            if a[0] == "c":
                key = (st, a[2])
                if buf_key is not None and buf_key != key: flush()
                buf_key = key; buf.append(a[1]); last_rpr = a[2]
            else:
                if (item, st) in emitted: continue
                emitted.add((item, st)); flush()
                p.append(copy.deepcopy(a[1]) if st == "eq" else self._del_opaque(a[1]))
            if item in anchors and (item, "anc") not in emitted:
                emitted.add((item, "anc")); flush()
                for m in anchors[item]: p.append(m)
        flush()
        return True

    # ------------------------------------------------------------------ paragraph insert/delete
    def _mark_para(self, p, kind):
        ppr = p.find(q("pPr"))
        if ppr is None: ppr = etree.Element(q("pPr")); p.insert(0, ppr)
        rpr = ppr.find(q("rPr"))
        if rpr is None: rpr = etree.SubElement(ppr, q("rPr"))
        for x in rpr.findall(q("ins")) + rpr.findall(q("del")): rpr.remove(x)
        rpr.insert(0, self.mark(kind))

    def new_para(self, template, text, rpr=None):
        """Paragraf baru (belum disisipkan) memakai pPr template, ditandai inserted."""
        p = etree.Element(q("p"))
        ppr = template.find(q("pPr"))
        if ppr is not None:
            ppr = copy.deepcopy(ppr)
            for x in ppr.findall(q("sectPr")) + ppr.findall(q("pPrChange")): ppr.remove(x)
            p.append(ppr)
        if rpr is None:
            r0 = next((r for r in template.iter(q("r")) if r.find(q("t")) is not None), None)
            rpr = r0.find(q("rPr")) if r0 is not None else None
        self._mark_para(p, "ins")
        if text: p.append(self._wrap("ins", [self._run(text, rpr)]))
        return p

    def insert_after(self, ref, *els):
        for el in els:
            ref.addnext(el); ref = el
        return ref

    def delete_para(self, p):
        for c in [c for c in p if c.tag != q("pPr")]:
            p.remove(c)
            if c.tag == q("r"):
                for t in c.findall(q("t")): t.tag = q("delText")
                p.append(self._wrap("del", [c]))
            elif c.tag == q("sdt"):
                p.append(self._del_opaque(c))
            else:
                p.append(c)
        self._mark_para(p, "del")

    # ------------------------------------------------------------------ tables
    def _mark_row(self, tr, kind):
        trpr = tr.find(q("trPr"))
        if trpr is None:
            trpr = etree.Element(q("trPr")); tblpr_ex = tr.find(q("tblPrEx"))
            (tblpr_ex.addnext(trpr) if tblpr_ex is not None else tr.insert(0, trpr))
        trpr.append(self.mark(kind))

    def delete_table(self, tbl):
        for tr in tbl.findall(q("tr")):
            self._mark_row(tr, "del")
            for p in tr.iter(q("p")): self.delete_para(p)

    def set_cell(self, tc, text, kind="ins"):
        ps = tc.findall(q("p"))
        for extra in ps[1:]: tc.remove(extra)
        p = ps[0]
        r0 = next((r for r in p.iter(q("r")) if r.find(q("t")) is not None), None)
        rpr = r0.find(q("rPr")) if r0 is not None else None
        for c in [c for c in p if c.tag != q("pPr")]: p.remove(c)
        self._mark_para(p, kind)
        if text: p.append(self._wrap(kind, [self._run(text, rpr)]))

    def new_table(self, template, rows, widths_dxa):
        """Tabel baru (semua baris inserted) dari template: baris 0 = gaya header, baris 1 = gaya isi."""
        tbl = copy.deepcopy(template)
        trs = tbl.findall(q("tr")); hdr, body = trs[0], trs[1]
        for tr in trs: tbl.remove(tr)
        grid = tbl.find(q("tblGrid"))
        for g in list(grid): grid.remove(g)
        for wd in widths_dxa:
            g = etree.SubElement(grid, q("gridCol")); g.set(q("w"), str(wd))
        tblw = tbl.find(q("tblPr")).find(q("tblW"))
        if tblw is not None: tblw.set(q("w"), str(sum(widths_dxa))); tblw.set(q("type"), "dxa")
        for ri, data in enumerate(rows):
            tr = copy.deepcopy(hdr if ri == 0 else body)
            for x in tr.findall(q("trPr")): tr.remove(x)
            tcs = tr.findall(q("tc"))
            for x in tcs: tr.remove(x)
            for ci, val in enumerate(data):
                tc = copy.deepcopy(tcs[0] if ci == 0 else tcs[min(ci, len(tcs) - 1)])
                tcpr = tc.find(q("tcPr"))
                for x in tcpr.findall(q("gridSpan")) + tcpr.findall(q("vMerge")): tcpr.remove(x)
                tcw = tcpr.find(q("tcW"))
                if tcw is not None: tcw.set(q("w"), str(widths_dxa[ci])); tcw.set(q("type"), "dxa")
                self.set_cell(tc, val)
                tr.append(tc)
            self._mark_row(tr, "ins")
            tbl.append(tr)
        return tbl

    def insert_row_after(self, tr_ref, values):
        tr = copy.deepcopy(tr_ref)
        for x in tr.findall(q("trPr")): tr.remove(x)
        for tc, v in zip(tr.findall(q("tc")), values): self.set_cell(tc, v)
        self._mark_row(tr, "ins")
        tr_ref.addnext(tr)
        return tr

    # ------------------------------------------------------------------ images
    def replace_image(self, p, new_rid, new_size_px, docpr_id, min_width_emu=0):
        """Gambar lama di p ditandai deleted; salinan drawing dengan blip baru ditandai inserted."""
        run = next(r for r in p.findall(q("r")) if r.find(q("drawing")) is not None)
        new = copy.deepcopy(run)
        new.find(".//{%s}blip" % A).set(f"{{{R}}}embed", new_rid)
        for crop in new.iter("{%s}srcRect" % A):          # crop gambar lama tidak berlaku untuk gambar baru
            for k in list(crop.attrib): del crop.attrib[k]
        ext = new.find(".//{%s}extent" % WP); cx = max(int(ext.get("cx")), min_width_emu)
        cy = int(cx * new_size_px[1] / new_size_px[0]); ext.set("cx", str(cx))
        ext.set("cy", str(cy))
        for x in new.iter("{%s}ext" % A):
            if x.get("cx"): x.set("cx", str(cx)); x.set("cy", str(cy))
        dp = new.find(".//{%s}docPr" % WP); dp.set("id", str(docpr_id)); dp.set("name", f"Picture {docpr_id}")
        i = p.index(run); p.remove(run)
        p.insert(i, self._wrap("del", [run]))
        p.insert(i + 1, self._wrap("ins", [new]))
