from __future__ import annotations

from typing import Any

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - import-time guard for local startup
    raise SystemExit("Gradio is not installed. Install with: pip install -e '.[gradio]' ") from exc

from debug import config
from debug.runner import InferenceRequest, RUNNER


DISPLAY_CHOICES = ["overlay", "heatmap"]
MODE_CHOICES = ["text-to-image", "fill-inpaint", "fill-outpaint"]


def _mode_updates(mode: str):
    is_t2i = mode == "text-to-image"
    is_fill = not is_t2i
    is_outpaint = mode == "fill-outpaint"
    guidance_value = config.DEFAULT_T2I_GUIDANCE if is_t2i else config.DEFAULT_FILL_GUIDANCE
    return (
        gr.update(visible=is_t2i),
        gr.update(visible=is_fill),
        gr.update(visible=is_fill),
        gr.update(visible=is_fill),
        gr.update(visible=is_fill),
        gr.update(visible=is_outpaint),
        gr.update(visible=is_fill),
        gr.update(value=guidance_value),
    )


def prepare_fill_canvas(
    mode: str,
    image_path: str,
    outpaint_left: int,
    outpaint_right: int,
    outpaint_top: int,
    outpaint_bottom: int,
    outpaint_overlap: int,
):
    if mode == "text-to-image":
        return None, None, None, None, None, "Fill canvas preparation is only used for fill modes."
    fill_state = RUNNER.prepare_fill_preview(
        mode=mode,
        image_path=image_path,
        expand_left=outpaint_left,
        expand_right=outpaint_right,
        expand_top=outpaint_top,
        expand_bottom=outpaint_bottom,
        overlap=outpaint_overlap,
    )
    status = (
        f"Prepared fill canvas: mode={mode} | size={fill_state['width']}x{fill_state['height']} | "
        "draw white mask regions in the editor before generating."
    )
    return (
        fill_state,
        fill_state["source_image"],
        fill_state["canvas_image"],
        fill_state["base_mask"],
        fill_state["canvas_image"],
        status,
    )


def run_generation(
    mode: str,
    prompt: str,
    width: int,
    height: int,
    num_steps: int,
    guidance: float,
    seed_text: str,
    image_path: str,
    outpaint_left: int,
    outpaint_right: int,
    outpaint_top: int,
    outpaint_bottom: int,
    outpaint_overlap: int,
    fill_state: dict[str, Any] | None,
    editor_value: Any,
):
    seed_value = None
    if seed_text.strip():
        seed_value = int(seed_text.strip())

    request = InferenceRequest(
        mode=mode,
        prompt=prompt,
        width=int(width),
        height=int(height),
        num_steps=int(num_steps),
        guidance=float(guidance),
        seed=seed_value,
        image_path=image_path or None,
        outpaint_left=int(outpaint_left),
        outpaint_right=int(outpaint_right),
        outpaint_top=int(outpaint_top),
        outpaint_bottom=int(outpaint_bottom),
        outpaint_overlap=int(outpaint_overlap),
    )
    result = RUNNER.run(request=request, fill_state=fill_state, editor_value=editor_value)
    head_choices = result["head_choices"]
    return (
        result["run_state"],
        result["generated_image"],
        result["default_view"]["overlay"],
        result["default_view"]["summary"],
        result["run_state"]["run_dir"],
        gr.update(choices=result["token_choices"], value=result["default_token_labels"]),
        gr.update(choices=result["run_state"]["layer_choices"], value=result["default_layer"]),
        gr.update(choices=result["run_state"]["step_choices"], value=result["default_step"]),
        gr.update(choices=head_choices, value=result["default_head"], visible=bool(head_choices)),
    )


def render_selection(
    run_state: dict[str, Any] | None,
    layer_label: str,
    step_label: str,
    token_labels: list[str],
    head_label: str | None,
    display_mode: str,
):
    if run_state is None:
        raise gr.Error("Run one inference first.")
    view = RUNNER.render_saved_view(
        run_state=run_state,
        layer_label=layer_label,
        step_label=step_label,
        token_labels=token_labels,
        head_label=head_label,
    )
    display_image = view[display_mode]
    summary = f"{view['summary']} | heatmap={view['heatmap_path']} | overlay={view['overlay_path']}"
    return display_image, summary


with gr.Blocks(title="FLUX Debug WebUI") as demo:
    gr.Markdown("# FLUX Debug / Visualization WebUI")
    gr.Markdown(
        "Raw attention view is strictly `image-query × text-key` from double-stream blocks only. "
        "T5 tokens are shown at subtoken granularity; CLIP is used only as a global conditioning vector."
    )

    fill_state = gr.State(value=None)
    run_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(MODE_CHOICES, value=config.DEFAULT_MODEL_MODE, label="Mode")
            prompt = gr.Textbox(value=config.DEFAULT_PROMPT, label="Prompt", lines=4)
            with gr.Group(visible=config.DEFAULT_MODEL_MODE == "text-to-image") as t2i_group:
                width = gr.Slider(128, 2048, value=config.DEFAULT_WIDTH, step=16, label="Width")
                height = gr.Slider(128, 2048, value=config.DEFAULT_HEIGHT, step=16, label="Height")
            with gr.Group(visible=config.DEFAULT_MODEL_MODE != "text-to-image") as fill_group:
                image_path = gr.Textbox(label="Local image path", placeholder="D:\\images\\input.png")
                prepare_button = gr.Button("Prepare Fill Canvas")
            with gr.Group(visible=config.DEFAULT_MODEL_MODE == "fill-outpaint") as outpaint_group:
                outpaint_left = gr.Number(value=config.DEFAULT_OUTPAINT_LEFT, label="Expand Left (px)")
                outpaint_right = gr.Number(value=config.DEFAULT_OUTPAINT_RIGHT, label="Expand Right (px)")
                outpaint_top = gr.Number(value=config.DEFAULT_OUTPAINT_TOP, label="Expand Top (px)")
                outpaint_bottom = gr.Number(value=config.DEFAULT_OUTPAINT_BOTTOM, label="Expand Bottom (px)")
                outpaint_overlap = gr.Number(value=config.DEFAULT_OUTPAINT_OVERLAP, label="Overlap (px)")
            num_steps = gr.Slider(1, 50, value=config.DEFAULT_NUM_STEPS, step=1, label="Number of steps")
            guidance = gr.Number(value=config.DEFAULT_T2I_GUIDANCE, label="Guidance")
            seed = gr.Textbox(value=str(config.DEFAULT_SEED), label="Seed (-1 for random)")
            generate_button = gr.Button("Generate")
            status_box = gr.Markdown()
            run_dir = gr.Textbox(label="Run directory", interactive=False)

        with gr.Column(scale=1):
            source_preview = gr.Image(label="Source Image", type="pil", visible=config.DEFAULT_MODEL_MODE != "text-to-image")
            canvas_preview = gr.Image(label="Canvas", type="pil", visible=config.DEFAULT_MODEL_MODE != "text-to-image")
            base_mask_preview = gr.Image(label="Base Mask", type="pil", visible=config.DEFAULT_MODEL_MODE != "text-to-image")
            mask_editor = gr.ImageEditor(label="Mask Editor", type="pil", visible=config.DEFAULT_MODEL_MODE != "text-to-image")

    with gr.Row():
        generated_image = gr.Image(label="Generated Image", type="pil")
        attention_view = gr.Image(label="Attention View", type="pil")

    with gr.Accordion("Attention Controls", open=True):
        token_selector = gr.CheckboxGroup(label="T5 tokens", choices=[])
        with gr.Row():
            layer_selector = gr.Dropdown(label="Layer", choices=[])
            step_selector = gr.Dropdown(label="Timestep", choices=[])
            head_selector = gr.Dropdown(label="Head", choices=[], visible=(config.HEAD_MODE == "all"))
            display_mode = gr.Radio(DISPLAY_CHOICES, value="overlay", label="Display")
        render_button = gr.Button("Render Selection")
        selection_summary = gr.Markdown()

    mode.change(
        _mode_updates,
        inputs=[mode],
        outputs=[t2i_group, fill_group, source_preview, canvas_preview, base_mask_preview, outpaint_group, mask_editor, guidance],
    )

    prepare_button.click(
        prepare_fill_canvas,
        inputs=[mode, image_path, outpaint_left, outpaint_right, outpaint_top, outpaint_bottom, outpaint_overlap],
        outputs=[fill_state, source_preview, canvas_preview, base_mask_preview, mask_editor, status_box],
    )

    generate_button.click(
        run_generation,
        inputs=[
            mode,
            prompt,
            width,
            height,
            num_steps,
            guidance,
            seed,
            image_path,
            outpaint_left,
            outpaint_right,
            outpaint_top,
            outpaint_bottom,
            outpaint_overlap,
            fill_state,
            mask_editor,
        ],
        outputs=[
            run_state,
            generated_image,
            attention_view,
            selection_summary,
            run_dir,
            token_selector,
            layer_selector,
            step_selector,
            head_selector,
        ],
    )

    render_button.click(
        render_selection,
        inputs=[run_state, layer_selector, step_selector, token_selector, head_selector, display_mode],
        outputs=[attention_view, selection_summary],
    )


if __name__ == "__main__":
    demo.launch()
