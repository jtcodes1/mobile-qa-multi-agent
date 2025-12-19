import xml.etree.ElementTree as ET


class ScreenReader:
    """Deterministic screen summarizer.

    We DON'T use an LLM for perception here because:
    - UI hierarchy dumps are cheaper + more stable than vision tokens
    - The LLM is better spent on planning + judging

    What we pass to the LLM:
    - screenshot path (so a human can verify in recordings)
    - wm size
    - a short list of UI nodes (labels + ids + bounds)
    """

    def __init__(self, device):
        self.device = device

    def describe(self, screenshot_path) -> str:
        parts = []
        parts.append(f"Screenshot: {screenshot_path}")
        parts.append(f"wm size: {self.device.wm_size()}")

        xml = self.device.ui_dump()
        if not xml:
            parts.append("UI dump: (empty/unavailable)")
            return "\n".join(parts)

        parts.append("UI dump: available")
        try:
            root = ET.fromstring(xml)
        except Exception:
            parts.append("UI dump parse: failed (non-XML)")
            return "\n".join(parts)

        nodes = []
        focusable_inputs = []

        for node in root.iter("node"):
            text = (node.attrib.get("text") or "").strip()
            desc = (node.attrib.get("content-desc") or "").strip()
            rid = (node.attrib.get("resource-id") or "").strip()
            clickable = (node.attrib.get("clickable") == "true")
            focusable = (node.attrib.get("focusable") == "true")
            enabled = (node.attrib.get("enabled") != "false")
            bounds = (node.attrib.get("bounds") or "").strip()

            if not enabled:
                continue

            if focusable and (rid or text or desc):
                focusable_inputs.append((rid, text, desc, bounds))

            # Keep nodes that are likely useful to an agent.
            if clickable or text or desc or rid:
                nodes.append((clickable, focusable, text, desc, rid, bounds))

        # Helpful trick:
        # Show focusable fields first because that's where typing bugs happen.
        focusable_inputs = focusable_inputs[:15]
        if focusable_inputs:
            parts.append("Focusable fields:")
            for i, (rid, text, desc, bounds) in enumerate(focusable_inputs, start=1):
                label = text or desc or "(no label)"
                parts.append(f"  F{i:02d}. {label} | id={rid or '-'} | bounds={bounds or '-'}")

        # Prefer clickable nodes, then nodes with labels.
        nodes.sort(key=lambda t: (not t[0], not bool(t[2] or t[3]), not t[1], t[4]))

        parts.append("Top UI nodes (clickable first):")
        max_nodes = 35
        for i, (clickable, focusable, text, desc, rid, bounds) in enumerate(nodes[:max_nodes], start=1):
            label = text or desc or "(no label)"
            flags = []
            if clickable:
                flags.append("clickable")
            if focusable:
                flags.append("focusable")
            flag_str = ",".join(flags) if flags else "-"
            parts.append(f"{i:02d}. {flag_str} | {label} | id={rid or '-'} | bounds={bounds or '-'}")

        return "\n".join(parts)
