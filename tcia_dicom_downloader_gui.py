import tkinter as tk
from tkinter import ttk, messagebox, Canvas
from tcia_utils import nbia
import threading
import os
import requests
import queue
# Add DICOM and image imports
import pydicom
from PIL import Image, ImageTk
from tkinter import filedialog
import sys

# Debug: Write command-line arguments to a file
try:
    with open("dicom_viewer_args.txt", "w") as f:
        f.write("Arguments: " + repr(sys.argv))
except Exception as e:
    print(f"Could not write dicom_viewer_args.txt: {e}")

# Helper to update scrollregion
def on_frame_configure(event):
    series_frame_canvas.configure(scrollregion=series_frame_canvas.bbox("all"))

# Helper to fetch dropdown options
def fetch_options():
    collections = [c['Collection'] for c in nbia.getCollections()]
    body_parts = [bp['BodyPartExamined'] for bp in nbia.getBodyPartExaminedValues()]
    modalities = [m['Modality'] for m in nbia.getModalityValues()]
    manufacturers = [m['Manufacturer'] for m in nbia.getManufacturerValues()]
    models = [m['ManufacturerModelName'] for m in nbia.getManufacturerModelNameValues()]
    return collections, body_parts, modalities, manufacturers, models

def setup_full_gui():
    global root, row, collection_var, body_part_var, modality_var, manufacturer_var, model_var, patient_id_var, study_uid_var
    global collection_cb, body_part_cb, modality_cb, manufacturer_cb, model_cb, patient_id_entry, study_uid_entry
    global search_btn, series_frame_canvas, series_scrollbar, series_inner_frame, series_inner_frame_id, series_vars
    global select_all_btn, global_series, download_btn, progress, view_dicom_btn, monai_infer_btn
    root = tk.Tk()
    root.title("TCIA DICOM Downloader")
    root.geometry("900x600")
    try:
        collection_options, body_part_options, modality_options, manufacturer_options, model_options = fetch_options()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch filter options: {e}")
        root.destroy()
        raise
    collection_var = tk.StringVar()
    body_part_var = tk.StringVar()
    modality_var = tk.StringVar()
    manufacturer_var = tk.StringVar()
    model_var = tk.StringVar()
    patient_id_var = tk.StringVar()
    study_uid_var = tk.StringVar()
    row = 0
    tk.Label(root, text="Collection:").grid(row=row, column=0, sticky="e")
    collection_cb = ttk.Combobox(root, textvariable=collection_var, values=collection_options, width=30)
    collection_cb.grid(row=row, column=1, sticky="w")
    row += 1
    tk.Label(root, text="Body Part:").grid(row=row, column=0, sticky="e")
    body_part_cb = ttk.Combobox(root, textvariable=body_part_var, values=body_part_options, width=30)
    body_part_cb.grid(row=row, column=1, sticky="w")
    row += 1
    tk.Label(root, text="Modality:").grid(row=row, column=0, sticky="e")
    modality_cb = ttk.Combobox(root, textvariable=modality_var, values=modality_options, width=30)
    modality_cb.grid(row=row, column=1, sticky="w")
    row += 1
    tk.Label(root, text="Manufacturer:").grid(row=row, column=0, sticky="e")
    manufacturer_cb = ttk.Combobox(root, textvariable=manufacturer_var, values=manufacturer_options, width=30)
    manufacturer_cb.grid(row=row, column=1, sticky="w")
    row += 1
    tk.Label(root, text="Manufacturer Model Name:").grid(row=row, column=0, sticky="e")
    model_cb = ttk.Combobox(root, textvariable=model_var, values=model_options, width=30)
    model_cb.grid(row=row, column=1, sticky="w")
    row += 1
    tk.Label(root, text="Patient ID:").grid(row=row, column=0, sticky="e")
    patient_id_entry = tk.Entry(root, textvariable=patient_id_var, width=33)
    patient_id_entry.grid(row=row, column=1, sticky="w")
    row += 1
    tk.Label(root, text="Study Instance UID:").grid(row=row, column=0, sticky="e")
    study_uid_entry = tk.Entry(root, textvariable=study_uid_var, width=33)
    study_uid_entry.grid(row=row, column=1, sticky="w")
    row += 1
    search_btn = tk.Button(root, text="Search Series")
    search_btn.grid(row=row, column=0, columnspan=2, pady=8)
    row += 1
    series_frame_canvas = Canvas(root, width=800, height=250)
    series_frame_canvas.grid(row=row, column=0, columnspan=3, pady=8, sticky='nsew')
    series_scrollbar = ttk.Scrollbar(root, orient="vertical", command=series_frame_canvas.yview)
    series_scrollbar.grid(row=row, column=3, sticky='ns')
    series_frame_canvas.configure(yscrollcommand=series_scrollbar.set)
    series_inner_frame = tk.Frame(series_frame_canvas)
    series_inner_frame_id = series_frame_canvas.create_window((0, 0), window=series_inner_frame, anchor='nw')
    row += 1
    series_vars = []
    select_all_btn = tk.Button(root, text="Select All", width=15)
    select_all_btn.grid(row=row, column=0, pady=4, sticky='w')
    series_inner_frame.bind("<Configure>", on_frame_configure)
    global_series = []
    search_btn.config(command=search_series)
    select_all_btn.config(command=select_all)
    row += 1
    download_btn = tk.Button(root, text="Download Selected Series")
    download_btn.grid(row=row, column=0, columnspan=2, pady=8)
    download_btn.config(command=download_selected_series)
    view_dicom_btn = tk.Button(root, text="View DICOM Image", command=open_and_display_dicom)
    view_dicom_btn.grid(row=row, column=0, columnspan=2, pady=8)
    row += 1
    monai_infer_btn = tk.Button(root, text="Run MONAI Inference on DICOM", command=run_monai_infer)
    monai_infer_btn.grid(row=row, column=0, columnspan=2, pady=8)
    progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
    progress.grid(row=row+1, column=0, columnspan=2, pady=8)
    root.mainloop()

def show_dicom_only(dicom_path):
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    import pydicom
    from PIL import Image, ImageTk
    import sys

    root = tk.Tk()
    root.title("DICOM Viewer")
    root.geometry("420x460")

    def on_close():
        try:
            root.destroy()
        except Exception:
            pass
        try:
            root.quit()
        except Exception:
            pass
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        im = Image.fromarray(img)
        if im.mode != "L":
            im = im.convert("L")
        im = im.resize((400, 400))
    except Exception as e:
        try:
            messagebox.showerror("Error", f"Failed to open DICOM file: {e}")
        except Exception:
            pass
        on_close()
        return

    tk_img = ImageTk.PhotoImage(im)

    TOOL_RECT = 'Rectangle'
    TOOL_ELLIPSE = 'Ellipse'
    TOOL_PENCIL = 'Pencil'
    TOOL_TEXT = 'Text'
    tool = tk.StringVar(value=TOOL_RECT)

    toolbar = tk.Frame(root)
    toolbar.pack(side='top', fill='x')
    rect_btn = tk.Button(toolbar, text='Rectangle', relief='sunken', command=lambda: set_tool(TOOL_RECT))
    ellipse_btn = tk.Button(toolbar, text='Ellipse', command=lambda: set_tool(TOOL_ELLIPSE))
    pencil_btn = tk.Button(toolbar, text='Pencil', command=lambda: set_tool(TOOL_PENCIL))
    text_btn = tk.Button(toolbar, text='Text', command=lambda: set_tool(TOOL_TEXT))
    rect_btn.pack(side='left', padx=2, pady=2)
    ellipse_btn.pack(side='left', padx=2, pady=2)
    pencil_btn.pack(side='left', padx=2, pady=2)
    text_btn.pack(side='left', padx=2, pady=2)

    canvas = tk.Canvas(root, width=400, height=400, bg='black')
    canvas.pack(padx=10, pady=10)
    canvas.create_image(0, 0, anchor='nw', image=tk_img)

    # Annotation state
    rect = None
    ellipse = None
    start_x = 0
    start_y = 0
    rectangles = []  # List of (id, coords)
    ellipses = []   # List of (id, coords)
    selected_rect = None
    selected_ellipse = None
    pencil_line = None
    pencil_points = []
    pencil_lines = []  # List of line ids
    text_items = []  # List of (id, text)
    text_entry = None
    rect_handles = []  # List of handle ids for selected rectangle
    move_mode = False
    resize_mode = False
    resize_handle_index = None
    move_offset = (0, 0)
    ellipse_handles = []  # List of handle ids for selected ellipse
    ellipse_move_mode = False
    ellipse_resize_mode = False
    ellipse_resize_handle_index = None
    ellipse_move_offset = (0, 0)

    HANDLE_SIZE = 8

    def draw_rect_handles(rect_coords):
        # Remove old handles
        for h in rect_handles:
            canvas.delete(h)
        rect_handles.clear()
        x0, y0, x1, y1 = rect_coords
        # 4 corners
        handle_coords = [
            (x0, y0), (x1, y0), (x1, y1), (x0, y1)
        ]
        for (hx, hy) in handle_coords:
            h = canvas.create_rectangle(hx-HANDLE_SIZE//2, hy-HANDLE_SIZE//2, hx+HANDLE_SIZE//2, hy+HANDLE_SIZE//2, fill='white', outline='black', tags='handle')
            rect_handles.append(h)

    def point_in_rect(x, y, coords):
        x0, y0, x1, y1 = coords
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        return x0 <= x <= x1 and y0 <= y <= y1

    def handle_hit_test(x, y, coords):
        # Returns index of handle if (x, y) is inside a handle, else None
        x0, y0, x1, y1 = coords
        handle_coords = [
            (x0, y0), (x1, y0), (x1, y1), (x0, y1)
        ]
        for i, (hx, hy) in enumerate(handle_coords):
            if abs(x-hx) <= HANDLE_SIZE and abs(y-hy) <= HANDLE_SIZE:
                return i
        return None

    def update_selected_rect_coords(new_coords):
        # Update rectangle and its stored coords
        canvas.coords(selected_rect, *new_coords)
        for i, (rid, coords) in enumerate(rectangles):
            if rid == selected_rect:
                rectangles[i] = (rid, list(new_coords))
                break
        draw_rect_handles(new_coords)

    def set_tool(selected):
        tool.set(selected)
        # Update button relief
        rect_btn.config(relief='sunken' if selected == TOOL_RECT else 'raised')
        ellipse_btn.config(relief='sunken' if selected == TOOL_ELLIPSE else 'raised')
        pencil_btn.config(relief='sunken' if selected == TOOL_PENCIL else 'raised')
        text_btn.config(relief='sunken' if selected == TOOL_TEXT else 'raised')
        # Deselect any selected annotation and reset move/resize state
        deselect_all()
        reset_move_resize_state()
        update_bindings()

    def reset_move_resize_state():
        nonlocal move_mode, resize_mode, resize_handle_index, move_offset
        nonlocal ellipse_move_mode, ellipse_resize_mode, ellipse_resize_handle_index, ellipse_move_offset
        move_mode = False
        resize_mode = False
        resize_handle_index = None
        move_offset = (0, 0)
        ellipse_move_mode = False
        ellipse_resize_mode = False
        ellipse_resize_handle_index = None
        ellipse_move_offset = (0, 0)

    def update_bindings():
        # Unbind all
        canvas.unbind('<ButtonPress-1>')
        canvas.unbind('<B1-Motion>')
        canvas.unbind('<ButtonRelease-1>')
        canvas.unbind('<Button-3>')
        if tool.get() == TOOL_RECT:
            canvas.bind('<ButtonPress-1>', on_mouse_down)
            canvas.bind('<B1-Motion>', on_mouse_drag)
            canvas.bind('<ButtonRelease-1>', on_mouse_up)
            canvas.bind('<Button-3>', on_right_click_rect)
        elif tool.get() == TOOL_ELLIPSE:
            canvas.bind('<ButtonPress-1>', on_mouse_down)
            canvas.bind('<B1-Motion>', on_mouse_drag)
            canvas.bind('<ButtonRelease-1>', on_mouse_up)
            canvas.bind('<Button-3>', on_right_click_ellipse)
        elif tool.get() == TOOL_PENCIL:
            canvas.bind('<ButtonPress-1>', on_mouse_down)
        elif tool.get() == TOOL_TEXT:
            canvas.bind('<ButtonPress-1>', on_mouse_down)

    def deselect_all():
        nonlocal selected_rect, selected_ellipse
        if selected_rect:
            canvas.itemconfig(selected_rect, outline='red')
            for h in rect_handles:
                canvas.delete(h)
            rect_handles.clear()
            selected_rect = None
        if selected_ellipse:
            canvas.itemconfig(selected_ellipse, outline='green')
            for h in ellipse_handles:
                canvas.delete(h)
            ellipse_handles.clear()
            selected_ellipse = None
        reset_move_resize_state()

    def on_mouse_down(event):
        nonlocal rect, ellipse, start_x, start_y, selected_rect, selected_ellipse, move_mode, resize_mode, resize_handle_index, move_offset
        nonlocal ellipse_move_mode, ellipse_resize_mode, ellipse_resize_handle_index, ellipse_move_offset
        # Removed event.num check to ensure left-click always works
        if tool.get() == TOOL_RECT:
            # If a rectangle is selected, check for move/resize
            if selected_rect:
                coords = canvas.coords(selected_rect)
                hidx = handle_hit_test(event.x, event.y, coords)
                if hidx is not None:
                    print('Rectangle: Start resize')
                    resize_mode = True
                    resize_handle_index = hidx
                    start_x, start_y = event.x, event.y
                    return
                if point_in_rect(event.x, event.y, coords):
                    print('Rectangle: Start move')
                    move_mode = True
                    move_offset = (event.x - coords[0], event.y - coords[1])
                    start_x, start_y = event.x, event.y
                    return
            # Otherwise, start new rect
            deselect_all()
            start_x, start_y = event.x, event.y
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)
        elif tool.get() == TOOL_ELLIPSE:
            if selected_ellipse:
                coords = canvas.coords(selected_ellipse)
                hidx = ellipse_handle_hit_test(event.x, event.y, coords)
                if hidx is not None:
                    print('Ellipse: Start resize')
                    ellipse_resize_mode = True
                    ellipse_resize_handle_index = hidx
                    start_x, start_y = event.x, event.y
                    return
                if point_in_rect(event.x, event.y, coords):
                    print('Ellipse: Start move')
                    ellipse_move_mode = True
                    ellipse_move_offset = (event.x - coords[0], event.y - coords[1])
                    start_x, start_y = event.x, event.y
                    return
            deselect_all()
            start_x, start_y = event.x, event.y
            ellipse = canvas.create_oval(start_x, start_y, start_x, start_y, outline='green', width=2)
        elif tool.get() == TOOL_PENCIL:
            pencil_points.clear()
            pencil_points.append((event.x, event.y))
            pencil_line = canvas.create_line(event.x, event.y, event.x, event.y, fill='yellow', width=2, smooth=True)
        elif tool.get() == TOOL_TEXT:
            text = simpledialog.askstring("Text", "Enter text to place:")
            if text:
                text_id = canvas.create_text(event.x, event.y, text=text, fill='cyan', anchor='nw', font=('Arial', 14, 'bold'))
                text_items.append((text_id, text))

    def on_mouse_drag(event):
        nonlocal rect, ellipse, move_mode, resize_mode, resize_handle_index, move_offset
        nonlocal ellipse_move_mode, ellipse_resize_mode, ellipse_resize_handle_index, ellipse_move_offset
        if tool.get() == TOOL_RECT:
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)
            elif selected_rect and move_mode:
                coords = canvas.coords(selected_rect)
                dx = event.x - start_x
                dy = event.y - start_y
                new_coords = [coords[0]+dx, coords[1]+dy, coords[2]+dx, coords[3]+dy]
                update_selected_rect_coords(new_coords)
                start_x, start_y = event.x, event.y
            elif selected_rect and resize_mode:
                coords = canvas.coords(selected_rect)
                x0, y0, x1, y1 = coords
                # 0: topleft, 1: topright, 2: bottomright, 3: bottomleft
                if resize_handle_index == 0:
                    x0, y0 = event.x, event.y
                elif resize_handle_index == 1:
                    x1, y0 = event.x, event.y
                elif resize_handle_index == 2:
                    x1, y1 = event.x, event.y
                elif resize_handle_index == 3:
                    x0, y1 = event.x, event.y
                new_coords = [x0, y0, x1, y1]
                update_selected_rect_coords(new_coords)
        elif tool.get() == TOOL_ELLIPSE:
            if ellipse:
                canvas.coords(ellipse, start_x, start_y, event.x, event.y)
            elif selected_ellipse and ellipse_move_mode:
                coords = canvas.coords(selected_ellipse)
                dx = event.x - start_x
                dy = event.y - start_y
                new_coords = [coords[0]+dx, coords[1]+dy, coords[2]+dx, coords[3]+dy]
                update_selected_ellipse_coords(new_coords)
                start_x, start_y = event.x, event.y
            elif selected_ellipse and ellipse_resize_mode:
                coords = canvas.coords(selected_ellipse)
                x0, y0, x1, y1 = coords
                if ellipse_resize_handle_index == 0:
                    x0, y0 = event.x, event.y
                elif ellipse_resize_handle_index == 1:
                    x1, y0 = event.x, event.y
                elif ellipse_resize_handle_index == 2:
                    x1, y1 = event.x, event.y
                elif ellipse_resize_handle_index == 3:
                    x0, y1 = event.x, event.y
                new_coords = [x0, y0, x1, y1]
                update_selected_ellipse_coords(new_coords)
        elif tool.get() == TOOL_PENCIL:
            if pencil_line:
                pencil_points.append((event.x, event.y))
                canvas.coords(pencil_line, *sum(pencil_points, ()))

    def on_mouse_up(event):
        nonlocal rect, ellipse, pencil_line, move_mode, resize_mode, resize_handle_index
        nonlocal ellipse_move_mode, ellipse_resize_mode, ellipse_resize_handle_index
        if tool.get() == TOOL_RECT:
            if rect:
                rectangles.append((rect, canvas.coords(rect)))
                rect = None
            move_mode = False
            resize_mode = False
            resize_handle_index = None
        elif tool.get() == TOOL_ELLIPSE:
            if ellipse:
                ellipses.append((ellipse, canvas.coords(ellipse)))
                ellipse = None
            ellipse_move_mode = False
            ellipse_resize_mode = False
            ellipse_resize_handle_index = None
        elif tool.get() == TOOL_PENCIL:
            if pencil_line:
                pencil_lines.append(pencil_line)
                pencil_line = None

    def on_right_click_rect(event):
        nonlocal selected_rect
        if tool.get() == TOOL_RECT:
            for rid, coords in rectangles:
                x0, y0, x1, y1 = coords
                if x0 > x1: x0, x1 = x1, x0
                if y0 > y1: y0, y1 = y1, y0
                if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                    deselect_all()
                    selected_rect = rid
                    canvas.itemconfig(rid, outline='blue')
                    draw_rect_handles(canvas.coords(rid))
                    break
            else:
                deselect_all()

    def on_right_click_ellipse(event):
        nonlocal selected_ellipse
        if tool.get() == TOOL_ELLIPSE:
            for eid, coords in ellipses:
                x0, y0, x1, y1 = coords
                if x0 > x1: x0, x1 = x1, x0
                if y0 > y1: y0, y1 = y1, y0
                if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                    deselect_all()
                    selected_ellipse = eid
                    canvas.itemconfig(eid, outline='blue')
                    draw_ellipse_handles(canvas.coords(eid))
                    break
            else:
                deselect_all()

    def on_delete(event):
        nonlocal selected_rect, selected_ellipse
        if selected_rect:
            canvas.delete(selected_rect)
            for h in rect_handles:
                canvas.delete(h)
            rect_handles.clear()
            rectangles[:] = [(rid, coords) for rid, coords in rectangles if rid != selected_rect]
            selected_rect = None
        if selected_ellipse:
            canvas.delete(selected_ellipse)
            for h in ellipse_handles:
                canvas.delete(h)
            ellipse_handles.clear()
            ellipses[:] = [(eid, coords) for eid, coords in ellipses if eid != selected_ellipse]
            selected_ellipse = None

    update_bindings()
    root.bind('<Delete>', on_delete)

    root.mainloop()

# Search series logic
def search_series():
    search_btn.config(state=tk.DISABLED)
    # Clear previous checkboxes
    for widget in series_inner_frame.winfo_children():
        widget.destroy()
    series_vars.clear()
    global global_series
    filters = {}
    if collection_var.get():
        filters['collection'] = collection_var.get()
    if body_part_var.get():
        filters['bodyPartExamined'] = body_part_var.get()
    if modality_var.get():
        filters['modality'] = modality_var.get()
    if manufacturer_var.get():
        filters['manufacturer'] = manufacturer_var.get()
    if model_var.get():
        filters['manufacturerModelName'] = model_var.get()
    if patient_id_var.get():
        filters['patientID'] = patient_id_var.get()
    if study_uid_var.get():
        filters['studyInstanceUID'] = study_uid_var.get()
    def worker():
        global global_series
        try:
            series = nbia.getSeries(**filters)
            global_series = series
            if not series:
                tk.Label(series_inner_frame, text="No series found for the selected filters.").pack(anchor='w')
            else:
                for i, s in enumerate(series):
                    desc = f"{s.get('SeriesDescription', '')} | SeriesUID: {s['SeriesInstanceUID']} | PatientID: {s.get('PatientID', '')} | StudyUID: {s.get('StudyInstanceUID', '')} | Modality: {s.get('Modality', '')} | BodyPart: {s.get('BodyPartExamined', '')} | Manufacturer: {s.get('Manufacturer', '')} | Images: {s.get('ImageCount', '')}"
                    var = tk.BooleanVar()
                    cb = tk.Checkbutton(series_inner_frame, text=desc, variable=var, anchor='w', width=120, wraplength=750, justify='left')
                    cb.pack(anchor='w', padx=2, pady=1)
                    series_vars.append(var)
        except Exception as e:
            tk.Label(series_inner_frame, text=f"Error: {e}").pack(anchor='w')
        finally:
            search_btn.config(state=tk.NORMAL)
    threading.Thread(target=worker).start()

# Select All button logic
def select_all():
    for var in series_vars:
        var.set(True)

# Update download logic to download all checked series
def download_selected_series():
    print("Download button pressed")
    print(f"global_series length: {len(global_series)}")
    print(f"series_vars: {[var.get() for var in series_vars]}")
    selected_indices = [i for i, var in enumerate(series_vars) if var.get()]
    print(f"selected_indices: {selected_indices}")
    if not selected_indices or not global_series:
        print("No series selected or no global_series")
        messagebox.showwarning("No selection", "Please select at least one series to download.")
        return
    output_dir = os.path.join(os.getcwd(), "dicom_download")
    print(f"Creating output directory at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    progress['value'] = 0
    download_btn.config(state=tk.DISABLED)
    progress_queue = queue.Queue()
    def worker():
        try:
            for idx in selected_indices:
                series_uid = global_series[idx]['SeriesInstanceUID']
                print(f"Starting download for series {series_uid}")
                try:
                    size_info = nbia.getSeriesSize(seriesInstanceUID=series_uid)
                    num_images = size_info[0]['ObjectCount'] if size_info and 'ObjectCount' in size_info[0] else None
                except Exception:
                    num_images = None
                url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_uid}"
                local_zip = os.path.join(output_dir, f"{series_uid}.zip")
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        total = int(r.headers.get('content-length', 0))
                        downloaded = 0
                        chunk_size = 8192
                        with open(local_zip, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total:
                                        percent = int(downloaded / total * 100)
                                        progress_queue.put(percent)
                    print(f"Download finished for {series_uid}, file at {local_zip}")
                    if not os.path.exists(local_zip):
                        print(f"ERROR: File {local_zip} was not created!")
                        progress_queue.put(('error', f"File {local_zip} was not created!"))
                    else:
                        progress_queue.put(100)
                        progress_queue.put(('done', series_uid, local_zip))
                except Exception as e:
                    print(f"Exception during download: {e}")
                    progress_queue.put(('error', str(e)))
            progress_queue.put('all_done')
        except Exception as e:
            print(f"Outer exception: {e}")
            progress_queue.put(('error', str(e)))
    def update_progress():
        try:
            while True:
                item = progress_queue.get_nowait()
                if isinstance(item, int):
                    progress['value'] = item
                elif isinstance(item, tuple) and item[0] == 'done':
                    _, series_uid, local_zip = item
                    messagebox.showinfo("Download complete", f"Downloaded series {series_uid} to {local_zip}\n\nImages are saved as a ZIP file in the 'dicom_download' folder in your current working directory.")
                elif item == 'all_done':
                    download_btn.config(state=tk.NORMAL)
                    return
                elif isinstance(item, tuple) and item[0] == 'error':
                    messagebox.showerror("Error", f"Failed to download series: {item[1]}")
                    download_btn.config(state=tk.NORMAL)
                    return
        except queue.Empty:
            pass
        root.after(100, update_progress)
    threading.Thread(target=worker, daemon=True).start()
    update_progress()

def open_and_display_dicom():
    file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")])
    if not file_path:
        return
    try:
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array
        # Normalize and convert to PIL Image
        im = Image.fromarray(img)
        if im.mode != "L":
            im = im.convert("L")
        im = im.resize((400, 400))  # Resize for display
        tk_img = ImageTk.PhotoImage(im)
        # If label already exists, update it; else, create it
        if hasattr(open_and_display_dicom, 'img_label'):
            open_and_display_dicom.img_label.config(image=tk_img)
            open_and_display_dicom.img_label.image = tk_img
        else:
            open_and_display_dicom.img_label = tk.Label(root, image=tk_img)
            open_and_display_dicom.img_label.image = tk_img
            open_and_display_dicom.img_label.grid(row=row+2, column=0, columnspan=2, pady=8)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open DICOM file: {e}")

# After GUI setup and before root.mainloop()
# If a file path is provided as a command-line argument, open and display it
if len(sys.argv) > 1:
    dicom_path = sys.argv[1]
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        im = Image.fromarray(img)
        if im.mode != "L":
            im = im.convert("L")
        im = im.resize((400, 400))
        tk_img = ImageTk.PhotoImage(im)
        if hasattr(open_and_display_dicom, 'img_label'):
            open_and_display_dicom.img_label.config(image=tk_img)
            open_and_display_dicom.img_label.image = tk_img
        else:
            open_and_display_dicom.img_label = tk.Label(root, image=tk_img)
            open_and_display_dicom.img_label.image = tk_img
            open_and_display_dicom.img_label.grid(row=row+2, column=0, columnspan=2, pady=8)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open DICOM file: {e}")

# MONAI Label inference function

def monai_label_infer(image_path, model="deepedit", server_url="http://localhost:8000"):
    """
    Send a DICOM image to the MONAI Label server for inference.
    Args:
        image_path (str): Path to the DICOM file.
        model (str): Model name (e.g., 'deepedit', 'segmentation', etc.).
        server_url (str): Base URL of the MONAI Label server.
    Returns:
        dict: The JSON response from the server.
    """
    infer_url = f"{server_url}/infer/{model}"
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(infer_url, files=files)
    response.raise_for_status()
    return response.json()

# Add this function after open_and_display_dicom

def run_monai_infer():
    file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")])
    if not file_path:
        return
    try:
        result = monai_label_infer(file_path, model="deepedit", server_url="http://localhost:8000")
        messagebox.showinfo("MONAI Inference Result", f"Result for {os.path.basename(file_path)}:\n{result}")
    except Exception as e:
        messagebox.showerror("MONAI Inference Error", f"Failed to run inference:\n{e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        show_dicom_only(sys.argv[1])
    else:
        setup_full_gui() 