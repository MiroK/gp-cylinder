from dolfin import Function, dof_to_vertex_map, warning, Mesh, MeshEditor
import xml.etree.ElementTree as ET
from itertools import dropwhile
from mpi4py import MPI
import numpy as np


def data_reordering(V):
    '''Reshaping/reordering data read from files'''
    # HDF5/VTK store 3d vectors and 3d tensor so we need to chop the data
    # also reorder as in 2017.2.0 only(?) vertex values are dumped
    if V.ufl_element().value_shape() == ():
        dof2v = dof_to_vertex_map(V)
        reorder = lambda a: a[dof2v]

        return reorder
    
    Vi = V.sub(0).collapse()
    dof2v = dof_to_vertex_map(Vi)
    gdim = V.mesh().geometry().dim()
    # WARNING: below there are assumption on component ordering
    # Vector
    if len(V.ufl_element().value_shape()) == 1:
        # Ellim Z for vectors in 2d
        keep = [0, 1] if gdim == 2 else range(gdim)

        reorder = lambda a, keep=keep, dof2f=dof2v:(

            np.column_stack([row[dof2v] for row in (a[:, keep]).T]).flatten()
        )

        return reorder
    
    # And tensor
    if len(V.ufl_element().value_shape()) == 2:
        # Ellim Z
        keep = [0, 1, 3, 4] if gdim == 2 else range(gdim**2)
            
        reorder = lambda a, keep=keep, dof2f=dof2v:(
            np.column_stack([row[dof2v] for row in (a[:, keep]).T]).flatten()
        )

        return reorder

    
def read_vtu_function(vtus, V):
    '''Read in functions in V from VTUs files'''
    # NOTE: this would much easier with (py)vtk but that is not part of
    # the FEniCS stack so ...
    gdim = V.mesh().geometry().dim()
    assert gdim > 1
    
    if isinstance(vtus, str): vtus = [vtus]

    reorder = data_reordering(V)

    npoints, ncells = V.mesh().num_vertices(), V.mesh().num_cells()
    functions = []
    for vtu in vtus:
        f = Function(V)
        
        data = read_vtu_point_data(vtu, npoints, ncells)
        f.vector().set_local(reorder(data))

        functions.append(f)
    return functions


def read_vtu_point_data(vtu, nvertices, ncells):
    '''PointData element of ASCII VTU file'''
    tree = ET.parse(vtu)
    root = tree.getroot()
    grid = next(iter(root))
    piece = next(iter(grid))

    # Check consistency of mesh (somewhat)
    assert nvertices == int(piece.attrib['NumberOfPoints'])
    assert ncells == int(piece.attrib['NumberOfCells'])
    # Throw StopIteration
    point_data_elm = next(dropwhile(lambda x: x.tag != 'PointData', piece))
    data = next(iter(point_data_elm))

    ncomps = int(data.attrib.get('NumberOfComponents', 0))
    values = np.array(map(float, filter(bool, data.text.split(' '))))
    # Reshape for reorder (so it is same as H5File
    if ncomps:
        values = values.reshape((-1, ncomps))
    return values


def read_vtu_mesh(path, cell_type):
    '''Read in mesh from function stored in vtu file'''
    tree = ET.parse(path)
    root = tree.getroot()
    grid = next(iter(root))
    piece = next(iter(grid))

    points, cells, _ = list(piece)
    
    # Parse points
    point_data = next(iter(points))
    # Always 3d gdim with this file format
    gdim = cell_type.geometric_dimension()
    point_data = np.array(map(float, filter(bool, point_data.text.split(' '))))
    point_data = point_data.reshape((-1, 3))[:, :gdim]

    # Parse cells
    cell_data = next(iter(cells))
    cell_data = np.array(map(int, filter(bool, cell_data.text.split(' '))))
    cell_data = cell_data.reshape((-1, cell_type.num_vertices()))

    return make_mesh(point_data, cell_data, cell_type)


def make_mesh(vertices, cells, cell_type):
    '''Mesh from data by MeshEditor'''
    gdim = cell_type.geometric_dimension()
    assert vertices.shape[1] == gdim

    tdim = cell_type.topological_dimension()

    mesh = Mesh()
    editor = MeshEditor()

    editor.open(mesh, str(cell_type), tdim, gdim)            

    editor.init_vertices(len(vertices))
    editor.init_cells(len(cells))

    for vi, x in enumerate(vertices): editor.add_vertex(vi, x)

    for ci, c in enumerate(cells): editor.add_cell(ci, *c)
    
    editor.close()

    return mesh
