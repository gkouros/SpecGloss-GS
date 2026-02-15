import time
import viser
import viser.transforms as tf
import numpy as np

host = "0.0.0.0"
port = 8080

server = {
    "server": None,
    "render_type": None,
    "client": None,
}

def init(initial_value='Rendered'):
    """Start the ViserServer and return immediately.
    All client setup happens in on_client_connect."""
    global server

    # Create and configure the server
    server["server"] = viser.ViserServer(host=host, port=port)
    server["server"].scene.world_axes.visible = True
    server["server"].scene.set_up_direction(direction='+z')

    @server["server"].on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        """Fires whenever a new client connects."""
        server["client"] = client

        # 1) Show the client ID
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

        # 2) Add the Render Type dropdown
        server["render_type"] = client.gui.add_dropdown(
            "Render Type",
            options=[
                "Rendered",
                "gt image",
                "envmap",
                "envmap2",
                "alpha map",
                "diffuse color",
                "specular color",
                "residual color",
                "base color",
                "roughness",
                "metallic",
                "specular",
                "visibility",
                "direct light",
                "indirect light",
                "render normal",
                "surf normal",
                "surf depth",
                "distortion",
                "bounding volume map",
                "instance map",
                "stable normals",
                "stable delight diffuse",
                "feature map",
            ],
            initial_value=initial_value
        )

        # 3) Add and wire up the Reset Up button
        gui_reset_up = client.gui.add_button(
            "Reset up direction",
            hint="Align the camera’s up to the current view-up.",
        )
        @gui_reset_up.on_click
        def _reset_up(event: viser.GuiEvent) -> None:
            cl = event.client
            assert cl is not None
            cl.camera.up_direction = tf.SO3(cl.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

        # 4) Now that the client is ready, start your render or update loop
        start_render_loop(client)

    # If ViserServer needs an explicit “start” call, uncomment this:
    # server["server"].start()

    return server

def start_render_loop(client: viser.ClientHandle):
    """Example render loop: push frames whenever you like."""
    import threading

    def _loop():
        while True:
            # Read current GUI selection
            render_mode = server["render_type"].value if server["render_type"] else "Rendered"

            # TODO: render your scene based on render_mode
            # e.g. client.scene.set_render_mode(render_mode)

            time.sleep(1/30)  # 30 Hz update

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()

def on_gui_change():
    """Fallback accessor if you need the dropdown value elsewhere."""
    if server["render_type"] is None:
        return ""
    return server["render_type"].value
