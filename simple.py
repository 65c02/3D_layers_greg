from PIL import Image, ImageQt
import sys
import os

# Check for PyQt5
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                                 QSpinBox, QScrollArea, QGridLayout, QFrame,
                                 QListWidget, QListWidgetItem, QAbstractItemView)
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtCore import Qt, QSize
    import math
except ImportError:
    print("PyQt5 is required. Please install it with: pip install PyQt5")
    sys.exit(1)

def load_and_convert_to_palette(image_path, num_colors=256):
    """
    Charge une image, la convertit en mode palette avec un nombre spécifique de couleurs,
    et renvoie l'objet image.
    """
    try:
        img = Image.open(image_path)
        # Convertir en mode 'P' (Palette) avec le nombre de couleurs spécifié
        palette_img = img.quantize(colors=num_colors)
        return palette_img
    except Exception as e:
        print(f"Erreur lors du chargement ou de la conversion de l'image: {e}")
        return None

from PIL import Image, ImageQt, ImageChops

# ... (imports remain the same, just adding ImageChops if not present, but better to add it at top level if possible, or inside function)

def generate_palette_layers(palette_image, custom_order=None):
    """
    Prend une image en mode palette et génère les couches et masques pour chaque couleur.
    Utilise un masque cumulatif : 
    - Couche N est remplie de Couleur N.
    - Masque N interdit les pixels des couleurs 0..N-1.
    Renvoie une liste de dictionnaires : {'index': int, 'color': (r,g,b), 'layer': Image, 'mask': Image}
    """
    if palette_image.mode != 'P':
        print("L'image n'est pas en mode palette.")
        return []

    palette = palette_image.getpalette()
    used_colors = palette_image.getcolors(maxcolors=256)
    
    if not used_colors:
        return []

    # Trier par index de couleur pour un affichage cohérent
    # Si un ordre personnalisé est fourni, l'utiliser
    if custom_order:
        # Créer un mapping index -> position
        order_map = {idx: i for i, idx in enumerate(custom_order)}
        # Fonction de tri: si dans custom_order, utiliser sa position, sinon mettre à la fin trié par index
        def sort_key(item):
            idx = item[1]
            if idx in order_map:
                return (0, order_map[idx])
            else:
                return (1, idx)
        used_colors.sort(key=sort_key)
    else:
        used_colors.sort(key=lambda x: x[1])

    layers_data = []
    
    # Masque cumulatif : 0 = autorisé, 255 = interdit
    # Au début, tout est autorisé (tout noir)
    cumulative_mask = Image.new("L", palette_image.size, 0)

    for count, index in used_colors:
        if palette and len(palette) >= 3 * (index + 1):
            r = palette[3 * index]
            g = palette[3 * index + 1]
            b = palette[3 * index + 2]
        else:
            r, g, b = 0, 0, 0
        
        # Créer une image RGBA pleine de la couleur cible
        layer = Image.new("RGBA", palette_image.size, (r, g, b, 255))
        
        # Appliquer le masque cumulatif actuel
        # Là où cumulative_mask est 255 (interdit), l'alpha doit être 0.
        # Là où cumulative_mask est 0 (autorisé), l'alpha doit être 255.
        # Donc Alpha = Invert(cumulative_mask)
        alpha_mask = ImageChops.invert(cumulative_mask)
        layer.putalpha(alpha_mask)
        
        # Sauvegarder l'état actuel pour l'affichage
        layers_data.append({
            'index': index,
            'color': (r, g, b),
            'layer': layer,
            'mask': cumulative_mask.copy() # Copie car on va le modifier
        })
        
        # Mettre à jour le masque cumulatif pour la prochaine itération
        # On ajoute les pixels de la couleur courante aux interdits
        current_color_pixels = palette_image.point(lambda p: 255 if p == index else 0, mode='L')
        
        # On ajoute (union) les nouveaux pixels interdits au masque existant
        # 255 + 0 = 255, 255 + 255 = 255 (clamped), 0 + 0 = 0
        # ImageChops.lighter (Max) ou add fonctionnent ici car c'est binaire 0/255
        cumulative_mask = ImageChops.lighter(cumulative_mask, current_color_pixels)
        
    return layers_data

def save_layers_to_disk(layers_data, output_dir="sub_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for item in layers_data:
        index = item['index']
        layer = item['layer']
        filename = f"color_{index}.png"
        filepath = os.path.join(output_dir, filename)
        layer.save(filepath)
        print(f"Sauvegardé: {filepath}")

def save_to_obj(layers_data, output_path, z_scale=1.0, xy_scale=1.0):
    """
    Exports layers to an OBJ file with an associated MTL file for colors.
    Each layer is a separate object (o layer_n).
    """
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    dir_name = os.path.dirname(output_path)
    mtl_filename = base_name + ".mtl"
    mtl_path = os.path.join(dir_name, mtl_filename)
    
    if not os.path.exists(dir_name) and dir_name:
        os.makedirs(dir_name)

    # Write MTL file
    with open(mtl_path, 'w') as mtl_file:
        for item in layers_data:
            index = item['index']
            r, g, b = item['color']
            # Normalize colors to 0-1
            rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
            mtl_file.write(f"newmtl material_{index}\n")
            mtl_file.write(f"Kd {rn:.4f} {gn:.4f} {bn:.4f}\n")
            mtl_file.write("d 1.0\n") # Opaque
            mtl_file.write("illum 2\n\n")

    # Write OBJ file
    with open(output_path, 'w') as obj_file:
        obj_file.write(f"mtllib {mtl_filename}\n")
        
        vertex_offset = 1
        
        for i, item in enumerate(layers_data):
            index = item['index']
            layer = item['layer']
            width, height = layer.size
            data = layer.tobytes("raw", "RGBA")
            
            obj_file.write(f"o layer_{index}\n")
            obj_file.write(f"usemtl material_{index}\n")
            
            # Z-offset based on layer index (i)
            z = float(i) * z_scale
            
            # Scale factor for the cube depth
            zs = z_scale
            
            for y in range(height):
                for x in range(width):
                    pixel_idx = (y * width + x) * 4
                    a = data[pixel_idx + 3]
                    
                    if a > 0:
                        # Invert Y to match OpenGL/Image coords usually (0,0 is top-left in PIL, bottom-left in 3D usually)
                        # Let's keep it consistent with VoxelWidget: height - 1 - y
                        y_pos = height - 1 - y
                        
                        # Define 8 vertices for the cube at x, y_pos, z
                        # v x y z
                        # 0: x, y, z
                        # 1: x+1, y, z
                        # 2: x+1, y+1, z
                        # 3: x, y+1, z
                        # 4: x, y, z+1
                        # 5: x+1, y, z+1
                        # 6: x+1, y+1, z+1
                        # 7: x, y+1, z+1
                        
                        # Apply XY scaling
                        x0 = x * xy_scale
                        x1 = (x + 1) * xy_scale
                        y0 = y_pos * xy_scale
                        y1 = (y_pos + 1) * xy_scale

                        # Swap Y and Z for OBJ export (Y is up/thickness, Z is depth/image-y)
                        # v x y z -> v x z y
                        # z (stack) becomes y (height)
                        # y (image) becomes z (depth)
                        
                        obj_file.write(f"v {x0} {z} {y0}\n")
                        obj_file.write(f"v {x1} {z} {y0}\n")
                        obj_file.write(f"v {x1} {z} {y1}\n")
                        obj_file.write(f"v {x0} {z} {y1}\n")
                        obj_file.write(f"v {x0} {z+zs} {y0}\n")
                        obj_file.write(f"v {x1} {z+zs} {y0}\n")
                        obj_file.write(f"v {x1} {z+zs} {y1}\n")
                        obj_file.write(f"v {x0} {z+zs} {y1}\n")
                        
                        # Faces (1-based indices)
                        # Back: 0 1 2 3 (CCW from back)
                        # Front: 4 5 6 7 (CCW from front)
                        # Bottom: 0 1 5 4 (CCW from bottom)
                        # Top: 3 2 6 7 (CCW from top)
                        # Left: 0 4 7 3 (CCW from left)
                        # Right: 1 5 6 2 (CCW from right)
                        
                        vo = vertex_offset
                        obj_file.write(f"f {vo} {vo+1} {vo+2} {vo+3}\n") # Back
                        obj_file.write(f"f {vo+4} {vo+5} {vo+6} {vo+7}\n") # Front
                        obj_file.write(f"f {vo} {vo+1} {vo+5} {vo+4}\n") # Bottom
                        obj_file.write(f"f {vo+3} {vo+2} {vo+6} {vo+7}\n") # Top
                        obj_file.write(f"f {vo} {vo+4} {vo+7} {vo+3}\n") # Left
                        obj_file.write(f"f {vo+1} {vo+5} {vo+6} {vo+2}\n") # Right
                        
                        vertex_offset += 8

    print(f"Export OBJ terminé: {output_path}")

class ImageLabel(QLabel):
    """Custom Label to display image with aspect ratio preservation"""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #eee;")
        self.setMinimumSize(100, 100)

    def set_image(self, pil_image):
        if pil_image is None:
            self.clear()
            return
            
        # Ensure image is RGBA for consistent handling
        # This handles P, L, RGB, etc.
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
            
        # Convert to QImage manually to avoid ImageQt issues
        data = pil_image.tobytes("raw", "RGBA")
        qim = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
        
        # QImage references the data, so we must keep a reference to it if we were keeping the QImage.
        # But here we convert to QPixmap immediately, which makes a copy.
        pixmap = QPixmap.fromImage(qim)

        # Scale to fit label size while keeping aspect ratio
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.setPixmap(scaled_pixmap)
        self.original_pixmap = pixmap # Store for resizing

    def resizeEvent(self, event):
        if hasattr(self, 'original_pixmap') and self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QQuaternion, QVector3D

class VoxelWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers_data = []
        self.rotation = QQuaternion.fromAxisAndAngle(QVector3D(1.0, 0.0, 0.0), 30.0) * \
                        QQuaternion.fromAxisAndAngle(QVector3D(0.0, 1.0, 0.0), -45.0)
        self.zoom = -50.0
        self.z_scale = 1.0
        self.xy_scale = 1.0
        self.lastPos = None

    def set_layers(self, layers_data):
        self.layers_data = layers_data
        self.update()

    def recalculate_zoom(self):
        if not self.layers_data:
            return
            
        width, height = self.layers_data[0]['layer'].size
        num_layers = len(self.layers_data)
        
        # Calculate world dimensions
        # Calculate world dimensions
        # Swapped Y and Z:
        # Width (X) -> X
        # Height (Y_img) -> Z_world
        # Layers (Z_stack) -> Y_world
        w_world = width * self.xy_scale
        h_world = num_layers * self.z_scale # Height is now thickness
        d_world = height * self.xy_scale    # Depth is now image height
        
        # Calculate bounding sphere diameter (approx) or max dimension
        max_dim = max(w_world, h_world, d_world)
        
        # We want max_dim to occupy 80% of the view height
        # tan(fov/2) = (height/2) / dist
        # height_at_dist = 2 * dist * tan(22.5)
        # We want max_dim = 0.8 * height_at_dist
        # max_dim = 0.8 * 2 * dist * tan(22.5)
        # dist = max_dim / (1.6 * tan(22.5))
        
        fov_rad = 45.0 * math.pi / 180.0
        tan_half_fov = math.tan(fov_rad / 2.0)
        
        dist = max_dim / (1.6 * tan_half_fov)
        
        self.zoom = -dist
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        # Enable blending for transparency if needed, though cubes are solid
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.2, 0.2, 0.2, 1.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        
        # Apply Quaternion Rotation
        axis = QVector3D()
        angle = 0.0
        # getAxisAndAngle returns (axis, angle) in PyQt5
        # Note: angle is in degrees
        axis, angle = self.rotation.getAxisAndAngle()
        glRotatef(angle, axis.x(), axis.y(), axis.z())

        if not self.layers_data:
            return

        # Center the model
        width, height = self.layers_data[0]['layer'].size
        num_layers = len(self.layers_data)
        
        # Apply scaling FIRST
        # Scale (X, Y, Z) -> (XY, Thickness, XY)
        glScalef(self.xy_scale, self.z_scale, self.xy_scale)
        
        # Then translate to center
        # Center (X, Y, Z) -> (width/2, layers/2, height/2)
        glTranslatef(-width / 2.0, -num_layers / 2.0, -height / 2.0)

        # Draw Solids
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.render_voxels(draw_wireframe=False)
        
        # Draw Wireframe
        # Offset slightly to avoid z-fighting
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.5)
        self.render_voxels(draw_wireframe=True)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_POLYGON_OFFSET_LINE)

    def render_voxels(self, draw_wireframe=False):
        for i, item in enumerate(self.layers_data):
            layer = item['layer']
            width, height = layer.size
            data = layer.tobytes("raw", "RGBA")
            
            # Optimization: Use display lists if slow, but loop is fine for small images
            glBegin(GL_QUADS)
            for y in range(height):
                for x in range(width):
                    idx = (y * width + x) * 4
                    r = data[idx]
                    g = data[idx+1]
                    b = data[idx+2]
                    a = data[idx+3]
                    
                    if a > 0: # Visible pixel
                        if draw_wireframe:
                            # Black or Darker color for wireframe
                            glColor3f(0.0, 0.0, 0.0)
                        else:
                            glColor3ub(r, g, b)
                            
                        # Draw cube at (x, layer, y_img)
                        # x -> x
                        # y (stack) -> i
                        # z (depth) -> height - 1 - y
                        self.draw_cube(x, i, height - 1 - y)
            glEnd()

    def draw_cube(self, x, y, z):
        # Draw a unit cube at x, y, z
        # Front Face
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(x, y, z+1)
        glVertex3f(x+1, y, z+1)
        glVertex3f(x+1, y+1, z+1)
        glVertex3f(x, y+1, z+1)
        # Back Face
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(x, y, z)
        glVertex3f(x, y+1, z)
        glVertex3f(x+1, y+1, z)
        glVertex3f(x+1, y, z)
        # Top Face
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(x, y+1, z)
        glVertex3f(x, y+1, z+1)
        glVertex3f(x+1, y+1, z+1)
        glVertex3f(x+1, y+1, z)
        # Bottom Face
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(x, y, z)
        glVertex3f(x+1, y, z)
        glVertex3f(x+1, y, z+1)
        glVertex3f(x, y, z+1)
        # Right face
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(x+1, y, z)
        glVertex3f(x+1, y+1, z)
        glVertex3f(x+1, y+1, z+1)
        glVertex3f(x+1, y, z+1)
        # Left Face
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(x, y, z)
        glVertex3f(x, y, z+1)
        glVertex3f(x, y+1, z+1)
        glVertex3f(x, y+1, z)

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        glViewport((width - side) // 2, (height - side) // 2, side, side)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height, 0.1, 5000.0) # Increased zFar
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            # Quaternion Rotation
            # Drag X -> Rotate around Y axis
            # Drag Y -> Rotate around X axis
            # We want to rotate relative to the screen, so we apply rotation on the left
            
            # Sensitivity
            speed = 0.5
            
            # Create rotation for X movement (Yaw around Y axis)
            rot_y = QQuaternion.fromAxisAndAngle(QVector3D(0.0, 1.0, 0.0), dx * speed)
            
            # Create rotation for Y movement (Pitch around X axis)
            rot_x = QQuaternion.fromAxisAndAngle(QVector3D(1.0, 0.0, 0.0), dy * speed)
            
            # Apply rotations: new_rot = rot_x * rot_y * old_rot
            self.rotation = rot_x * rot_y * self.rotation
            self.rotation.normalize()
            
        elif event.buttons() & Qt.RightButton:
            # Adjust zoom speed based on current zoom to make it controllable
            self.zoom += dy * (abs(self.zoom) * 0.01 + 0.1)

        self.lastPos = event.pos()
        self.update()

    def wheelEvent(self, event):
        # Zoom with mouse wheel
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 0.9
        else:
            self.zoom *= 1.1
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Palette Layer Extractor & Voxel View")
        self.resize(1200, 900) # Reduced height

        self.current_image_path = None
        self.processed_layers = []

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls Area
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15) # Spacing between groups
        
        self.btn_load = QPushButton("Charger Image")
        self.btn_load.clicked.connect(self.load_image)
        controls_layout.addWidget(self.btn_load)

        def add_param(label, widget):
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0,0,0,0)
            l.setSpacing(2)
            l.addWidget(QLabel(label))
            l.addWidget(widget)
            controls_layout.addWidget(w)

        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(2, 256)
        self.spin_colors.setValue(4) # Default to 4 as requested
        add_param("Nb Couleurs:", self.spin_colors)

        from PyQt5.QtWidgets import QDoubleSpinBox
        self.spin_depth = QDoubleSpinBox()
        self.spin_depth.setRange(0.01, 5.0)
        self.spin_depth.setSingleStep(0.01)
        self.spin_depth.setValue(0.1) # Default to 0.1 fixed thickness
        self.spin_depth.valueChanged.connect(self.update_depth)
        add_param("Epaisseur Couche:", self.spin_depth)

        self.spin_scale_xy = QDoubleSpinBox()
        self.spin_scale_xy.setRange(0.001, 100.0)
        self.spin_scale_xy.setSingleStep(0.01)
        self.spin_scale_xy.setDecimals(4)
        self.spin_scale_xy.setValue(0.1)
        self.spin_scale_xy.valueChanged.connect(self.update_from_scale_xy)
        # add_param("Taille Pixel XY:", self.spin_scale_xy) # Hidden as requested

        self.spin_width_cm = QDoubleSpinBox()
        self.spin_width_cm.setRange(0.1, 1000.0)
        self.spin_width_cm.setSingleStep(1.0)
        self.spin_width_cm.setValue(10.0)
        self.spin_width_cm.valueChanged.connect(self.update_from_width_cm)
        add_param("Largeur (cm):", self.spin_width_cm)

        self.spin_height_cm = QDoubleSpinBox()
        self.spin_height_cm.setRange(0.1, 1000.0)
        self.spin_height_cm.setSingleStep(1.0)
        self.spin_height_cm.setValue(10.0)
        self.spin_height_cm.valueChanged.connect(self.update_from_height_cm)
        add_param("Hauteur (cm):", self.spin_height_cm)

        self.btn_process = QPushButton("Convertir & Extraire")
        self.btn_process.clicked.connect(self.process_image)
        controls_layout.addWidget(self.btn_process)
        


        self.btn_export_obj = QPushButton("Export Mesh")
        self.btn_export_obj.clicked.connect(self.export_obj)
        self.btn_export_obj.setEnabled(False)
        controls_layout.addWidget(self.btn_export_obj)

        main_layout.addLayout(controls_layout)

        # Image Display Area (Splitter or just VBox)
        # Top: Original / Converted
        top_display_layout = QHBoxLayout()
        
        self.lbl_original = ImageLabel()
        top_display_layout.addWidget(self.lbl_original, 1)
        
        self.lbl_converted = ImageLabel()
        top_display_layout.addWidget(self.lbl_converted, 1)
        
        main_layout.addLayout(top_display_layout, 1)

        # Bottom Area: Splitter for Layers List and 3D View
        from PyQt5.QtWidgets import QSplitter
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Layers List (Drag & Drop)
        self.layers_list = QListWidget()
        self.layers_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.layers_list.model().rowsMoved.connect(self.on_layers_reordered)
        
        layers_container = QWidget()
        layers_layout = QVBoxLayout(layers_container)
        layers_layout.addWidget(QLabel("Couches (Layers) et Masques:"))
        layers_layout.addWidget(self.layers_list)
        
        bottom_splitter.addWidget(layers_container)

        # Right: 3D Voxel View
        self.voxel_widget = VoxelWidget()
        voxel_container = QWidget()
        voxel_layout = QVBoxLayout(voxel_container)
        lbl_voxel = QLabel("Vue 3D Voxel:")
        lbl_voxel.setFixedHeight(int(lbl_voxel.fontMetrics().height() * 1.5))
        voxel_layout.addWidget(lbl_voxel)
        voxel_layout.addWidget(self.voxel_widget)
        
        bottom_splitter.addWidget(voxel_container)
        
        # Set initial sizes (50/50)
        bottom_splitter.setSizes([300, 900])

        main_layout.addWidget(bottom_splitter, 3) # Reduced stretch factor

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.current_image_path = path
            img = Image.open(path)
            self.lbl_original.set_image(img)
            self.lbl_converted.clear()
            self.clear_layers_grid()

            self.btn_export_obj.setEnabled(False)
            self.voxel_widget.set_layers([])

    def process_image(self):
        if not self.current_image_path:
            return

        num_colors = self.spin_colors.value()
        palette_img = load_and_convert_to_palette(self.current_image_path, num_colors)
        
        if palette_img:
            self.lbl_converted.set_image(palette_img)
            
            # Get current order if exists
            custom_order = []
            if self.layers_list.count() > 0:
                for i in range(self.layers_list.count()):
                    item = self.layers_list.item(i)
                    data = item.data(Qt.UserRole)
                    if data and 'index' in data:
                        custom_order.append(data['index'])

            # Generate layers
            self.processed_layers = generate_palette_layers(palette_img, custom_order)
            self.display_layers()
            self.voxel_widget.set_layers(self.processed_layers)
            self.update_depth(self.spin_depth.value())
            
            # Initialize dimensions based on image size and default xy_scale
            if self.processed_layers:
                width, height = self.processed_layers[0]['layer'].size
                # Default xy_scale is 0.1, so calculate cm
                # 0.1 unit = 1 cm => 1 unit = 10 cm
                # width_cm = width * xy_scale * 10
                current_xy = self.spin_scale_xy.value()
                w_cm = width * current_xy * 10.0
                h_cm = height * current_xy * 10.0
                
                self.spin_width_cm.setValue(w_cm)
                self.spin_height_cm.setValue(h_cm)
                self.block_signals_dimensions(False)
                
                self.update_voxel_scale_xy()
                self.voxel_widget.recalculate_zoom() # Auto-zoom after processing

            self.btn_export_obj.setEnabled(True)

    def block_signals_dimensions(self, block):
        self.spin_scale_xy.blockSignals(block)
        self.spin_width_cm.blockSignals(block)
        self.spin_height_cm.blockSignals(block)

    def update_depth(self, value):
        self.voxel_widget.z_scale = value
        self.voxel_widget.update()

    def update_voxel_scale_xy(self):
        self.voxel_widget.xy_scale = self.spin_scale_xy.value()
        self.voxel_widget.update()

    def update_from_scale_xy(self, value):
        if not self.processed_layers:
            return
        
        width, height = self.processed_layers[0]['layer'].size
        w_cm = width * value * 10.0
        h_cm = height * value * 10.0
        
        self.block_signals_dimensions(True)
        self.spin_width_cm.setValue(w_cm)
        self.spin_height_cm.setValue(h_cm)
        self.block_signals_dimensions(False)
        
        self.update_voxel_scale_xy()

    def update_from_width_cm(self, value):
        if not self.processed_layers:
            return
            
        width, height = self.processed_layers[0]['layer'].size
        aspect_ratio = height / width
        
        new_h_cm = value * aspect_ratio
        # xy_scale = width_cm / (width_px * 10)
        new_scale = value / (width * 10.0)
        
        self.block_signals_dimensions(True)
        self.spin_height_cm.setValue(new_h_cm)
        self.spin_scale_xy.setValue(new_scale)
        self.block_signals_dimensions(False)
        
        self.update_voxel_scale_xy()

    def update_from_height_cm(self, value):
        if not self.processed_layers:
            return
            
        width, height = self.processed_layers[0]['layer'].size
        aspect_ratio = width / height
        
        new_w_cm = value * aspect_ratio
        # xy_scale = height_cm / (height_px * 10)
        new_scale = value / (height * 10.0)
        
        self.block_signals_dimensions(True)
        self.spin_width_cm.setValue(new_w_cm)
        self.spin_scale_xy.setValue(new_scale)
        self.block_signals_dimensions(False)
        
        self.update_voxel_scale_xy()

    def on_layers_reordered(self, parent, start, end, destination, row):
        # When layers are reordered, we want to regenerate everything (masks, 3D view)
        # using the new order. The easiest way is to call process_image, 
        # which reads the current list order.
        self.process_image()

    def clear_layers_grid(self):
        self.layers_list.clear()

    def display_layers(self):
        self.clear_layers_grid()
        
        for i, item in enumerate(self.processed_layers):
            # Create a custom widget for the item
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(5, 2, 5, 2)
            
            # Index
            layout.addWidget(QLabel(str(item['index'])))
            
            # Color Swatch
            r, g, b = item['color']
            swatch = QLabel()
            swatch.setFixedSize(30, 30)
            swatch.setStyleSheet(f"background-color: rgb({r},{g},{b}); border: 1px solid black;")
            layout.addWidget(swatch)
            
            # Layer Image
            layer_lbl = ImageLabel()
            layer_lbl.setFixedSize(40, 40)
            layer_lbl.set_image(item['layer'])
            layout.addWidget(layer_lbl)
            
            # Mask Image
            mask_lbl = ImageLabel()
            mask_lbl.setFixedSize(40, 40)
            mask_lbl.set_image(item['mask'])
            layout.addWidget(mask_lbl)
            
            layout.addStretch()
            
            # List Item
            list_item = QListWidgetItem(self.layers_list)
            list_item.setSizeHint(widget.sizeHint())
            list_item.setData(Qt.UserRole, item)
            
            self.layers_list.setItemWidget(list_item, widget)

    def export_obj(self):
        if self.processed_layers:
            # Default path: ./meshes/output.obj
            default_dir = os.path.join(os.getcwd(), "meshes")
            if not os.path.exists(default_dir):
                os.makedirs(default_dir)
            
            path, _ = QFileDialog.getSaveFileName(self, "Exporter OBJ", os.path.join(default_dir, "output.obj"), "OBJ Files (*.obj)")
            if path:
                save_to_obj(self.processed_layers, path, self.spin_depth.value(), self.spin_scale_xy.value())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
