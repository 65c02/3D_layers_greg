from PIL import Image, ImageQt
import sys
import os
from collections import defaultdict

CENTIMETRE_FACTOR = 1000.0
EPAISSEUR_FACTEUR = 10.0


# Check for PyQt5
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                                 QSpinBox, QScrollArea, QGridLayout, QFrame,
                                 QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox)
    from PyQt5.QtGui import QPixmap, QImage, QColor
    from PyQt5.QtCore import Qt, QSize, pyqtSignal
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

def save_to_obj(faces, output_path, z_scale=1.0, xy_scale=1.0):
    """
    Exports faces to an OBJ file with an associated MTL file for colors.
    """
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    dir_name = os.path.dirname(output_path)
    mtl_filename = base_name + ".mtl"
    mtl_path = os.path.join(dir_name, mtl_filename)
    
    if not os.path.exists(dir_name) and dir_name:
        os.makedirs(dir_name)

    # Collect unique colors for materials
    # Map color tuple (r,g,b) to material index
    materials = {}
    next_mat_id = 0
    
    for face in faces:
        color = face['color']
        if color not in materials:
            materials[color] = next_mat_id
            next_mat_id += 1

    # Write MTL file
    with open(mtl_path, 'w') as mtl_file:
        for color, mat_id in materials.items():
            r, g, b = color
            rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
            mtl_file.write(f"newmtl material_{mat_id}\n")
            mtl_file.write(f"Kd {rn:.4f} {gn:.4f} {bn:.4f}\n")
            mtl_file.write("d 1.0\n")
            mtl_file.write("illum 2\n\n")

    # Write OBJ file
    with open(output_path, 'w') as obj_file:
        obj_file.write(f"mtllib {mtl_filename}\n")
        
        vertex_offset = 1
        
        # Group faces by material to minimize usemtl calls (optional but good practice)
        # Or just write them as is. Let's write as is for simplicity or sort if needed.
        # Sorting by material might be better.
        faces.sort(key=lambda f: materials[f['color']])
        
        current_mat_id = -1
        
        for face in faces:
            # Material
            mat_id = materials[face['color']]
            if mat_id != current_mat_id:
                obj_file.write(f"usemtl material_{mat_id}\n")
                current_mat_id = mat_id
            
            # Vertices
            # Apply scaling
            # v = (x, y, z) -> (x*xy, y*z_scale, z*xy)
            # Note: y is stack (up), z is depth.
            # OBJ Y is Up.
            
            for v in face['vertices']:
                vx = v[0] * xy_scale
                vy = v[1] * z_scale
                vz = v[2] * xy_scale
                obj_file.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                
            # Normal
            nx, ny, nz = face['normal']
            obj_file.write(f"vn {nx} {ny} {nz}\n")
            
            # Face
            # f v1//vn v2//vn v3//vn v4//vn
            # Vertices are written in order, so they are vertex_offset, +1, +2, +3
            # Normal index is 1 (since we write one normal per face, wait.
            # Actually we write one normal per face? No, we write normals as needed.
            # But here we can just write the normal for the face and reference it.
            # Or simpler: we write the normal once?
            # Wait, `vn` indices are global.
            # If we write `vn` for every face, the index increments.
            
            # Let's write the normal for this face.
            # Normal index = face_index + 1
            # Vertex indices = vertex_offset ... vertex_offset+3
            
            # Actually, reusing normals is better, but writing one per face is easier.
            # Let's check if we can reuse standard normals.
            # We have 6 standard normals.
            # But let's just write it to be safe.
            
            # Actually, standard normals are better.
            # But my `generate_faces` returns a tuple.
            # Let's just write the normal index.
            # I'll write the normal for each face.
            
            vn_idx = vertex_offset // 4 + 1 # Rough approximation if we wrote 4 verts per face? No.
            # We write 1 normal per face.
            # So normal index is current face index + 1.
            # Wait, OBJ indices are 1-based global.
            # We are writing `vn` inside the loop.
            # So for face i (0-based):
            # We have written i normals before this one.
            # So this normal is index i+1.
            # We have written i*4 vertices before this one.
            # So vertices are i*4+1, i*4+2, i*4+3, i*4+4.
            
            # Wait, vertex_offset is tracked.
            
            # Correct logic:
            # We write 4 vertices.
            # We write 1 normal.
            # vn index is current global normal count.
            # v indices are current global vertex count.
            
            # We need to track global counts.
            # But since we write them sequentially:
            # Vertices: range(vertex_offset, vertex_offset+4)
            # Normal: face_index + 1 (if we write one per face)
            
            # Wait, I am writing `vn` inside the loop.
            # So the normal index is simply the number of `vn` lines written so far.
            # Which is `face_index + 1` if we enumerate.
            
            # But wait, I am iterating `faces`.
            # Let's just use a counter for normals too if I want to be explicit, or just calculate.
            
            # Actually, let's just write the normal and use relative indexing? No, standard OBJ uses absolute.
            
            # Let's use a counter.
            normal_index = (vertex_offset - 1) // 4 + 1
            
            obj_file.write(f"f {vertex_offset}//{normal_index} {vertex_offset+1}//{normal_index} {vertex_offset+2}//{normal_index} {vertex_offset+3}//{normal_index}\n")
            
            vertex_offset += 4

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

def generate_faces(layers_data, xy_scale=1.0, z_scale=1.0):
    """
    Generates a list of faces for the voxel model.
    Each face is a dict: {'vertices': [(x,y,z), ...], 'normal': (nx,ny,nz), 'color': (r,g,b)}
    """
    faces = []
    
    # Normals
    n_front = (0.0, 0.0, 1.0) # Z+
    n_back = (0.0, 0.0, -1.0) # Z-
    n_top = (0.0, 1.0, 0.0)   # Y+
    n_bottom = (0.0, -1.0, 0.0) # Y-
    n_right = (1.0, 0.0, 0.0) # X+
    n_left = (-1.0, 0.0, 0.0) # X-

    for i, item in enumerate(layers_data):
        layer = item['layer']
        width, height = layer.size
        data = layer.tobytes("raw", "RGBA")
        color = item['color']
        
        # Z-offset based on layer index (i)
        # In our 3D view:
        # x -> x
        # y (stack) -> i
        # z (depth) -> height - 1 - y_img
        
        # But wait, the VoxelWidget.draw_cube uses:
        # glVertex3f(x, y, z+1) where y is the stack index.
        # So Y is UP (stack).
        # X is Right.
        # Z is Depth (image Y).
        
        y_stack = float(i)
        
        for y_img in range(height):
            for x_img in range(width):
                idx = (y_img * width + x_img) * 4
                a = data[idx+3]
                
                if a > 0:
                    # Calculate coordinates
                    # x, y, z in the loop of draw_cube correspond to:
                    # x = x_img
                    # y = i (layer index)
                    # z = height - 1 - y_img
                    
                    x = float(x_img)
                    y = y_stack
                    z = float(height - 1 - y_img)
                    
                    # Vertices for the unit cube at (x, y, z)
                    # v0: x, y, z
                    # v1: x+1, y, z
                    # v2: x+1, y+1, z
                    # v3: x, y+1, z
                    # v4: x, y, z+1
                    # v5: x+1, y, z+1
                    # v6: x+1, y+1, z+1
                    # v7: x, y+1, z+1
                    
                    v0 = (x, y, z)
                    v1 = (x+1, y, z)
                    v2 = (x+1, y+1, z)
                    v3 = (x, y+1, z)
                    v4 = (x, y, z+1)
                    v5 = (x+1, y, z+1)
                    v6 = (x+1, y+1, z+1)
                    v7 = (x, y+1, z+1)
                    
                    # Add faces
                    # Front (Z+): v4, v5, v6, v7. Normal (0,0,1)
                    faces.append({'vertices': [v4, v5, v6, v7], 'normal': n_front, 'color': color})
                    
                    # Back (Z-): v0, v3, v2, v1. Normal (0,0,-1)
                    faces.append({'vertices': [v0, v3, v2, v1], 'normal': n_back, 'color': color})
                    
                    # Top (Y+): v3, v7, v6, v2. Normal (0,1,0)
                    faces.append({'vertices': [v3, v7, v6, v2], 'normal': n_top, 'color': color})
                    
                    # Bottom (Y-): v0, v1, v5, v4. Normal (0,-1,0)
                    faces.append({'vertices': [v0, v1, v5, v4], 'normal': n_bottom, 'color': color})
                    
                    # Right (X+): v1, v2, v6, v5. Normal (1,0,0)
                    faces.append({'vertices': [v1, v2, v6, v5], 'normal': n_right, 'color': color})
                    
                    # Left (X-): v0, v4, v7, v3. Normal (-1,0,0)
                    faces.append({'vertices': [v0, v4, v7, v3], 'normal': n_left, 'color': color})

    return faces

def optimize_faces(faces):
    """
    Removes duplicate faces.
    Two faces are considered duplicates if they have the same vertices (in any order).
    Since we are dealing with cubes, internal faces will appear twice (once for each cube sharing the face),
    but with opposite normals. However, if we just check vertex positions, they are the same polygon geometry.
    
    Wait, if two cubes share a face, the vertices are the same.
    Cube A (0,0,0) Right face: (1,0,0), (1,1,0), (1,1,1), (1,0,1)
    Cube B (1,0,0) Left face: (1,0,0), (1,0,1), (1,1,1), (1,1,0)
    
    The vertices are the same set of points.
    So we can sort the vertices of each face to create a key.
    If a key appears more than once (it should be exactly twice for internal faces), we remove BOTH.
    Because if it appears twice, it means it's an internal face between two opaque blocks.
    """
    face_groups = defaultdict(list)
    
    for face in faces:
        # Create a key from sorted vertices
        # Round to avoid float precision issues
        verts = sorted([tuple(round(c, 5) for c in v) for v in face['vertices']])
        key = tuple(verts)
        face_groups[key].append(face)
        
    optimized_faces = []
    removed_count = 0
    
    for key, group in face_groups.items():
        if len(group) == 1:
            optimized_faces.append(group[0])
        else:
            # If 2 or more, it's an internal face (or error), so we remove all instances
            # For a perfect grid of cubes, internal faces always come in pairs.
            removed_count += len(group)
            
    return optimized_faces, removed_count


class VoxelWidget(QOpenGLWidget):
    # Signal emitting (original_faces, optimized_faces, cube_count)
    meshOptimized = pyqtSignal(int, int, int) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers_data = []
        self.faces = [] # List of faces to render
        self.optimization_enabled = False
        
        self.rotation = QQuaternion.fromAxisAndAngle(QVector3D(1.0, 0.0, 0.0), 30.0) * \
                        QQuaternion.fromAxisAndAngle(QVector3D(0.0, 1.0, 0.0), -45.0)
        self.zoom = -50.0
        self.z_scale = 1.0
        self.xy_scale = 1.0
        self.lastPos = None

    def set_layers(self, layers_data):
        self.layers_data = layers_data
        self.update_mesh()

    def set_optimization(self, enabled):
        if self.optimization_enabled != enabled:
            self.optimization_enabled = enabled
            self.update_mesh()

    def update_mesh(self):
        if not self.layers_data:
            self.faces = []
            self.meshOptimized.emit(0, 0, 0)
            self.update()
            return

        # Generate full mesh
        faces = generate_faces(self.layers_data)
        original_count = len(faces)
        # Each cube has 6 faces, so cube count is faces / 6
        cube_count = original_count // 6
        
        optimized_count = original_count
        if self.optimization_enabled:
            faces, removed_count = optimize_faces(faces)
            optimized_count = len(faces)
            
        self.faces = faces
        self.meshOptimized.emit(original_count, optimized_count, cube_count)
        self.update()

    def recalculate_zoom(self):
        if not self.layers_data:
            return
            
        width, height = self.layers_data[0]['layer'].size
        num_layers = len(self.layers_data)
        
        # Calculate world dimensions
        w_world = width * self.xy_scale
        h_world = num_layers * self.z_scale
        d_world = height * self.xy_scale
        
        max_dim = max(w_world, h_world, d_world)
        
        fov_rad = 45.0 * math.pi / 180.0
        tan_half_fov = math.tan(fov_rad / 2.0)
        
        dist = max_dim / (1.6 * tan_half_fov)
        
        self.zoom = -dist
        self.update()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
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
        axis, angle = self.rotation.getAxisAndAngle()
        glRotatef(angle, axis.x(), axis.y(), axis.z())

        if not self.faces:
            return

        # Center the model
        # We need dimensions. Since we have faces, we can infer, or just use layers_data info
        if self.layers_data:
            width, height = self.layers_data[0]['layer'].size
            num_layers = len(self.layers_data)
            
            # Apply scaling FIRST
            glScalef(self.xy_scale, self.z_scale, self.xy_scale)
            
            # Then translate to center
            glTranslatef(-width / 2.0, -num_layers / 2.0, -height / 2.0)

        # Draw Solids
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.render_faces(draw_wireframe=False)
        
        # Draw Wireframe
        glEnable(GL_POLYGON_OFFSET_LINE)
        glPolygonOffset(-1.0, -1.0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glLineWidth(1.5)
        self.render_faces(draw_wireframe=True)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glDisable(GL_POLYGON_OFFSET_LINE)

    def render_faces(self, draw_wireframe=False):
        glBegin(GL_QUADS)
        for face in self.faces:
            if draw_wireframe:
                glColor3f(0.0, 0.0, 0.0)
            else:
                r, g, b = face['color']
                glColor3ub(r, g, b)
            
            # Normal
            nx, ny, nz = face['normal']
            glNormal3f(nx, ny, nz)
            
            # Vertices
            for v in face['vertices']:
                glVertex3f(v[0], v[1], v[2])
        glEnd()

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        glViewport((width - side) // 2, (height - side) // 2, side, side)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height, 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            speed = 0.5
            rot_y = QQuaternion.fromAxisAndAngle(QVector3D(0.0, 1.0, 0.0), dx * speed)
            rot_x = QQuaternion.fromAxisAndAngle(QVector3D(1.0, 0.0, 0.0), dy * speed)
            self.rotation = rot_x * rot_y * self.rotation
            self.rotation.normalize()
            
        elif event.buttons() & Qt.RightButton:
            self.zoom += dy * (abs(self.zoom) * 0.01 + 0.1)

        self.lastPos = event.pos()
        self.update()

    def wheelEvent(self, event):
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
        add_param("Epaisseur Couche (cm):", self.spin_depth)

        self.spin_scale_xy = QDoubleSpinBox()
        self.spin_scale_xy.setRange(0.001, 10000.0)
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
        
        # Optimization Checkbox
        self.chk_optimization = QCheckBox("Optimization")
        self.chk_optimization.setChecked(True)
        self.chk_optimization.stateChanged.connect(self.on_optimization_changed)
        controls_layout.addWidget(self.chk_optimization)

        self.btn_export_obj = QPushButton("Export Mesh")
        self.btn_export_obj.clicked.connect(self.export_obj)
        self.btn_export_obj.setEnabled(False)
        controls_layout.addWidget(self.btn_export_obj)

        main_layout.addLayout(controls_layout)

        # Image Display Area (Splitter or just VBox)
        # Top: Original / Stats (Removed converted image view)
        top_display_layout = QHBoxLayout()
        
        self.lbl_original = ImageLabel()
        top_display_layout.addWidget(self.lbl_original, 1)
        
        # Stats Label where converted image was
        self.lbl_stats = QLabel("Stats: -")
        self.lbl_stats.setAlignment(Qt.AlignCenter)
        self.lbl_stats.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; border: 1px solid #ccc; background-color: #eee;")
        top_display_layout.addWidget(self.lbl_stats, 1)
        
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
        self.voxel_widget.meshOptimized.connect(self.update_stats) # Connect signal
        
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
            self.clear_layers_grid()

            self.btn_export_obj.setEnabled(False)
            self.voxel_widget.set_layers([])
            self.lbl_stats.setText("Stats: -")

            # Update color count based on image
            # Try to count unique colors up to 256
            colors = img.getcolors(maxcolors=256)
            if colors:
                self.spin_colors.setValue(len(colors))
            else:
                self.spin_colors.setValue(256)

    def process_image(self):
        if not self.current_image_path:
            return

        num_colors = self.spin_colors.value()
        palette_img = load_and_convert_to_palette(self.current_image_path, num_colors)
        
        if palette_img:
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
            
            # Update Voxel Widget
            # Ensure optimization state is set
            self.voxel_widget.optimization_enabled = self.chk_optimization.isChecked()
            self.voxel_widget.set_layers(self.processed_layers)
            self.update_depth(self.spin_depth.value())
            
            # Initialize dimensions based on image size and default xy_scale
            if self.processed_layers:
                width, height = self.processed_layers[0]['layer'].size
                # Default xy_scale is 0.1, so calculate cm
                # 0.1 unit = 1 cm => 1 unit = 10 cm
                # width_cm = width * xy_scale * 10
                current_xy = self.spin_scale_xy.value()
                w_cm = width * current_xy / CENTIMETRE_FACTOR
                h_cm = height * current_xy / CENTIMETRE_FACTOR

                self.spin_width_cm.setValue(w_cm)
                self.spin_height_cm.setValue(h_cm)
                self.block_signals_dimensions(False)
                
                self.update_voxel_scale_xy()
                self.voxel_widget.recalculate_zoom() # Auto-zoom after processing

            self.btn_export_obj.setEnabled(True)

    def on_optimization_changed(self, state):
        self.voxel_widget.set_optimization(state == Qt.Checked)

    def update_stats(self, original, optimized, cubes):
        text = f"Cubes (Pixels): {cubes}\nPolygones: {original} -> {optimized}"
        self.lbl_stats.setText(text)

    def export_obj(self):
        if not self.voxel_widget.faces:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Sauvegarder OBJ", "", "OBJ Files (*.obj)")
        if path:
            # Use the faces from the voxel widget (which are already optimized if enabled)
            save_to_obj(self.voxel_widget.faces, path, 
                       z_scale=self.voxel_widget.z_scale, 
                       xy_scale=self.voxel_widget.xy_scale)

    def block_signals_dimensions(self, block):
        self.spin_scale_xy.blockSignals(block)
        self.spin_width_cm.blockSignals(block)
        self.spin_height_cm.blockSignals(block)

    def update_depth(self, value):
        self.voxel_widget.z_scale = value * EPAISSEUR_FACTEUR
        self.voxel_widget.update()

    def update_voxel_scale_xy(self):
        self.voxel_widget.xy_scale = self.spin_scale_xy.value()
        self.voxel_widget.update()

    def update_from_scale_xy(self, value):
        if not self.processed_layers:
            return
        
        width, height = self.processed_layers[0]['layer'].size
        w_cm = width * value * CENTIMETRE_FACTOR
        h_cm = height * value * CENTIMETRE_FACTOR
        
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
        new_scale = (value * CENTIMETRE_FACTOR) / width
        
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
        new_scale = (value * CENTIMETRE_FACTOR) / height 
        
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
            
            # Create a small pixmap for color
            pix = QPixmap(20, 20)
            pix.fill(QColor(r, g, b))
            lbl_color = QLabel()
            lbl_color.setPixmap(pix)
            layout.addWidget(lbl_color)
            
            # Info
            layout.addWidget(QLabel(f"RGB: {r},{g},{b}"))
            
            layout.addStretch()
            
            # Create List Item
            list_item = QListWidgetItem(self.layers_list)
            list_item.setSizeHint(widget.sizeHint())
            
            # Store data
            list_item.setData(Qt.UserRole, item)
            
            self.layers_list.setItemWidget(list_item, widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
