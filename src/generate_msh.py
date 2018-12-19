import os, subprocess, itertools
from numpy import deg2rad


def generate_mesh(args, template='geometry_2d.template_geo', dim=2):
    '''Modify template according args and make gmsh generate the mesh'''
    assert os.path.exists(template), template
    
    args = args.copy()

    with open(template, 'r') as f: old = f.readlines()

    # Chop the file to replace the jet positions
    split = map(lambda s: s.startswith('DefineConstant'), old).index(True)

    # Look for ] closing DefineConstant[
    # Make sure that args specifies all the constants (not necessary
    # as these have default values). This is more to check sanity of inputs
    last, _ = next(itertools.dropwhile(lambda (i, line): '];' not in line,
                                       enumerate(old)))
    constant_lines = old[split+1:last]
    constants = set(l.split('=')[0].strip() for l in constant_lines)

    jet_positions = deg2rad(map(float, args.pop('jet_positions')))
    jet_positions = 'jet_positions[] = {%s};\n' % (', '.join(map(str, jet_positions)))
    body = ''.join([jet_positions] + old[split:])

    output = args.pop('output')

    if not output:
        output = '_'.join([template, 'templateted.geo'])
    assert os.path.splitext(output)[1] == '.geo'

    with open(output, 'w') as f: f.write(body)

    args['jet_width'] = deg2rad(args['jet_width'])

    scale = args.pop('clscale')

    # What we think can be defined vs what can be
    assert set(args.keys()) <= constants, (set(args.keys())-constants)

    constant_values = ' '.join(['-setnumber %s %g' % item for item in args.items()])

    return subprocess.call(['gmsh -%d -clscale %g %s %s' % (dim, scale, constant_values, output)],
                           shell=True)

# -------------------------------------------------------------------

if __name__ == '__main__':
    import argparse, sys, petsc4py
    from math import pi

    parser = argparse.ArgumentParser(description='Generate msh file from GMSH',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optional output geo file
    parser.add_argument('-output', default='', type=str, help='A geofile for writing out geometry')
    # Geometry
    parser.add_argument('-length', default=200, type=float,
                        help='Channel length')
    parser.add_argument('-front_distance', default=40, type=float,
                        help='Cylinder center distance to inlet')

    parser.add_argument('-bottom_distance', default=40, type=float,
                        help='Cylinder center distance from bottom wall')
    parser.add_argument('-jet_radius', default=10, type=float,
                        help='Cylinder radius')
    parser.add_argument('-width', default=80, type=float,
                        help='Channel width')
    parser.add_argument('-cylinder_size', default=0.25, type=float,
                        help='Mesh size on cylinder')
    parser.add_argument('-box_size', default=5, type=float,
                        help='Mesh size on wall')
    # Jet perameters
    parser.add_argument('-jet_positions', nargs='+', default=[0, 60, 120, 180, 240, 300],
                        help='Angles of jet center points')
    parser.add_argument('-jet_width', default=10, type=float,
                        help='Jet width in degrees')

    # Refine geometry
    parser.add_argument('-clscale', default=1, type=float,
                        help='Scale the mesh size relative to give')

    args = parser.parse_args()

    # Using geometry_2d.geo to produce geometry_2d.msh
    sys.exit(generate_mesh(args.__dict__))
