import carla
import torch
import numpy as np
from PIL import Image
import cv2
from sklearn.linear_model import LinearRegression
from brains.CARLA.utils.ground_truth.camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    check_inside_image,
    create_lane_lines,
    get_matrix_global,
    CameraGeometry,
)
from collections import Counter

POINTS_PER_MAP = {
    'Carla/Maps/Town02_Opt': [{'location': {'x': 181.01332092285156, 'y': 302.49774169921875, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 178.749267578125, 'roll': 0.0}}, {'location': {'x': 161.05384826660156, 'y': 302.52716064453125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.92127990722656, 'roll': 0.0}}, {'location': {'x': 141.05386352539062, 'y': 302.55462646484375, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.92127990722656, 'roll': 0.0}}, {'location': {'x': 121.05730438232422, 'y': 302.5545654296875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 101.05730438232422, 'y': 302.5477600097656, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 81.05730438232422, 'y': 302.5409851074219, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 61.05730438232422, 'y': 302.5342102050781, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.01942443847656, 'roll': 0.0}}, {'location': {'x': 41.055267333984375, 'y': 302.53515625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -180.03903198242188, 'roll': 0.0}}, {'location': {'x': 21.055269241333008, 'y': 302.5487976074219, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.96095275878906, 'roll': 0.0}}, {'location': {'x': 1.594491720199585, 'y': 302.3045349121094, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 194.7761688232422, 'roll': 0.0}}, {'location': {'x': -3.3816475868225098, 'y': 287.3636474609375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.09098052978516, 'roll': 0.0}}, {'location': {'x': -3.413407802581787, 'y': 267.3636779785156, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.09098052978516, 'roll': 0.0}}, {'location': {'x': -3.440734386444092, 'y': 247.36642456054688, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.01293182373047, 'roll': 0.0}}, {'location': {'x': -3.4452500343322754, 'y': 227.36642456054688, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.01293182373047, 'roll': 0.0}}, {'location': {'x': -3.449765682220459, 'y': 207.36642456054688, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.01293182373047, 'roll': 0.0}}, {'location': {'x': 0.4272531569004059, 'y': 191.67459106445312, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -13.926624298095703, 'roll': 0.0}}, {'location': {'x': 19.930631637573242, 'y': 191.56138610839844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.02626265585422516, 'roll': 0.0}}, {'location': {'x': 38.929359436035156, 'y': 192.07168579101562, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 26.47684669494629, 'roll': 0.0}}, {'location': {'x': 41.85167694091797, 'y': 208.76499938964844, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.99503326416016, 'roll': 0.0}}, {'location': {'x': 41.90330123901367, 'y': 228.9360809326172, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 85.05287170410156, 'roll': 0.0}}, {'location': {'x': 58.411155700683594, 'y': 240.8790283203125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.0348052978515625, 'roll': 0.0}}, {'location': {'x': 78.41114807128906, 'y': 240.89117431640625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.0348052978515625, 'roll': 0.0}}, {'location': {'x': 98.41114044189453, 'y': 240.9033203125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.0348052978515625, 'roll': 0.0}}, {'location': {'x': 118.41114044189453, 'y': 240.91546630859375, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.0348052978515625, 'roll': 0.0}}, {'location': {'x': 138.4111328125, 'y': 240.9276123046875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -359.96514892578125, 'roll': 0.0}}, {'location': {'x': 158.4111328125, 'y': 240.93975830078125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.0348052978515625, 'roll': 0.0}}, {'location': {'x': 178.4111328125, 'y': 240.951904296875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.0348052978515625, 'roll': 0.0}}, {'location': {'x': 193.69517517089844, 'y': 229.41815185546875, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.01294708251953, 'roll': 0.0}}, {'location': {'x': 193.69065856933594, 'y': 209.41815185546875, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.01294708251953, 'roll': 0.0}}, {'location': {'x': 189.16549682617188, 'y': 188.97874450683594, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -148.71475219726562, 'roll': 0.0}}, {'location': {'x': 168.32078552246094, 'y': 187.62939453125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.02626037597656, 'roll': 0.0}}, {'location': {'x': 148.32078552246094, 'y': 187.62022399902344, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.02626037597656, 'roll': 0.0}}, {'location': {'x': 128.32078552246094, 'y': 187.61105346679688, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.02626037597656, 'roll': 0.0}}, {'location': {'x': 108.32078552246094, 'y': 187.60189819335938, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.02626037597656, 'roll': 0.0}}, {'location': {'x': 88.32078552246094, 'y': 187.5927276611328, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.02626037597656, 'roll': 0.0}}, {'location': {'x': 68.32079315185547, 'y': 187.58355712890625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.02626037597656, 'roll': 0.0}}, {'location': {'x': 47.657344818115234, 'y': 188.1053009033203, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 159.1720428466797, 'roll': 0.0}}],
    'Carla/Maps/Town06': [{'location': {'x': 181.01710510253906, 'y': 251.58737182617188, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 201.01710510253906, 'y': 251.5941925048828, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 221.01710510253906, 'y': 251.60101318359375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 241.01710510253906, 'y': 251.6078338623047, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 261.01708984375, 'y': 251.61465454101562, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 281.01708984375, 'y': 251.62147521972656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 301.01708984375, 'y': 251.62831115722656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 321.01708984375, 'y': 251.6351318359375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 341.0170593261719, 'y': 251.64195251464844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 361.0170593261719, 'y': 251.64877319335938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 381.0170593261719, 'y': 251.6555938720703, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 401.01708984375, 'y': 251.66241455078125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 421.01708984375, 'y': 251.6692352294922, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 441.01708984375, 'y': 251.6760711669922, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 461.0170593261719, 'y': 251.68289184570312, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 481.0170593261719, 'y': 251.68971252441406, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 501.0170593261719, 'y': 251.696533203125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 521.01708984375, 'y': 251.70335388183594, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 541.01708984375, 'y': 251.71017456054688, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 561.01708984375, 'y': 251.7169952392578, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 581.01708984375, 'y': 251.7238311767578, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 601.27294921875, 'y': 251.7080078125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -1.1779546737670898, 'roll': 0.0}}, {'location': {'x': 623.7069702148438, 'y': 248.78729248046875, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -13.65749740600586, 'roll': 0.0}}, {'location': {'x': 644.9798583984375, 'y': 241.08775329589844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -26.137041091918945, 'roll': 0.0}}, {'location': {'x': 664.636962890625, 'y': 223.1511993408203, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -58.77001953125, 'roll': 0.0}}, {'location': {'x': 671.4813232421875, 'y': 197.86082458496094, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.37858581542969, 'roll': 0.0}}, {'location': {'x': 671.6982421875, 'y': 177.86154174804688, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.62139892578125, 'roll': 0.0}}, {'location': {'x': 671.9151611328125, 'y': 157.8627166748047, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.62139892578125, 'roll': 0.0}}, {'location': {'x': 672.132080078125, 'y': 137.8638916015625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.62139892578125, 'roll': 0.0}}, {'location': {'x': 672.3489379882812, 'y': 117.86505889892578, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.62139892578125, 'roll': 0.0}}, {'location': {'x': 672.5658569335938, 'y': 97.86624145507812, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.62139892578125, 'roll': 0.0}}, {'location': {'x': 672.7827758789062, 'y': 77.86741638183594, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.62139892578125, 'roll': 0.0}}, {'location': {'x': 672.6138916015625, 'y': 57.00111389160156, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 266.5399475097656, 'roll': 0.0}}, {'location': {'x': 669.8013916015625, 'y': 35.495384216308594, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 258.55877685546875, 'roll': 0.0}}, {'location': {'x': 664.0302124023438, 'y': 14.588459014892578, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 250.57760620117188, 'roll': 0.0}}, {'location': {'x': 651.7012329101562, 'y': -6.960171222686768, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 227.0525360107422, 'roll': 0.0}}, {'location': {'x': 630.115234375, 'y': -20.986661911010742, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 198.9786834716797, 'roll': 0.0}}, {'location': {'x': 606.4669799804688, 'y': -23.919649124145508, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.41941833496094, 'roll': 0.0}}, {'location': {'x': 586.467529296875, 'y': -24.06605339050293, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.58056640625, 'roll': 0.0}}, {'location': {'x': 566.4680786132812, 'y': -24.21245574951172, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.58056640625, 'roll': 0.0}}, {'location': {'x': 546.4056396484375, 'y': -24.345693588256836, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.8748779296875, 'roll': 0.0}}, {'location': {'x': 526.4056396484375, 'y': -24.389362335205078, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.8748779296875, 'roll': 0.0}}, {'location': {'x': 506.40570068359375, 'y': -24.433032989501953, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.8748779296875, 'roll': 0.0}}, {'location': {'x': 486.40576171875, 'y': -24.476701736450195, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.8748779296875, 'roll': 0.0}}, {'location': {'x': 466.40582275390625, 'y': -24.52037239074707, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -179.8748779296875, 'roll': 0.0}}, {'location': {'x': 446.3643493652344, 'y': -24.561180114746094, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -180.06924438476562, 'roll': 0.0}}, {'location': {'x': 426.3346252441406, 'y': -24.489749908447266, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 406.3347473144531, 'y': -24.41690444946289, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 386.33489990234375, 'y': -24.344058990478516, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 366.33502197265625, 'y': -24.271215438842773, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 346.33514404296875, 'y': -24.19837188720703, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 326.3352966308594, 'y': -24.125526428222656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 306.33544921875, 'y': -24.052682876586914, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 286.3355712890625, 'y': -23.97983741760254, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 266.335693359375, 'y': -23.906993865966797, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 246.33584594726562, 'y': -23.834148406982422, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 226.33596801757812, 'y': -23.761302947998047, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 206.3361053466797, 'y': -23.688459396362305, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 186.33624267578125, 'y': -23.615615844726562, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 166.33636474609375, 'y': -23.542770385742188, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 146.33648681640625, 'y': -23.469924926757812, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 126.33662414550781, 'y': -23.397079467773438, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 106.33675384521484, 'y': -23.324234008789062, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 86.33688354492188, 'y': -23.25139045715332, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 66.33702087402344, 'y': -23.178546905517578, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 46.33715057373047, 'y': -23.105701446533203, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 26.3372859954834, 'y': -23.032855987548828, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': 10.151309967041016, 'y': -27.720903396606445, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -112.74484252929688, 'roll': 0.0}}, {'location': {'x': 9.779897689819336, 'y': -46.95023727416992, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.3822937011719, 'roll': 0.0}}, {'location': {'x': 9.913345336914062, 'y': -66.94979095458984, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.3822937011719, 'roll': 0.0}}, {'location': {'x': 10.046794891357422, 'y': -86.94934844970703, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 270.3822937011719, 'roll': 0.0}}, {'location': {'x': 10.276938438415527, 'y': -106.6629409790039, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -87.75643920898438, 'roll': 0.0}}, {'location': {'x': 14.087690353393555, 'y': -123.95154571533203, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -73.25007629394531, 'roll': 0.0}}, {'location': {'x': 14.501405715942383, 'y': -144.58480834960938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -121.717041015625, 'roll': 0.0}}, {'location': {'x': -5.01180362701416, 'y': -151.4821319580078, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 161.0759735107422, 'roll': 0.0}}, {'location': {'x': -16.401058197021484, 'y': -134.16470336914062, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 85.58792114257812, 'roll': 0.0}}, {'location': {'x': -12.346339225769043, 'y': -114.4681625366211, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 81.7371826171875, 'roll': 0.0}}, {'location': {'x': -11.901618957519531, 'y': -94.75387573242188, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 90.38229370117188, 'roll': 0.0}}, {'location': {'x': -12.03506851196289, 'y': -74.75431823730469, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 90.38229370117188, 'roll': 0.0}}, {'location': {'x': -12.168516159057617, 'y': -54.75476837158203, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 90.38229370117188, 'roll': 0.0}}, {'location': {'x': -12.30196475982666, 'y': -34.75521469116211, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 90.38229370117188, 'roll': 0.0}}, {'location': {'x': -21.175256729125977, 'y': -22.859804153442383, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -41.175506591796875, 'y': -22.786958694458008, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -61.17536926269531, 'y': -22.714115142822266, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -81.17523956298828, 'y': -22.64126968383789, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -101.17510986328125, 'y': -22.568424224853516, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -121.17497253417969, 'y': -22.49557876586914, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -141.1748504638672, 'y': -22.4227352142334, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -161.17471313476562, 'y': -22.349891662597656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -181.17457580566406, 'y': -22.27704620361328, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.7913055419922, 'roll': 0.0}}, {'location': {'x': -201.17445373535156, 'y': -22.204200744628906, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.79132080078125, 'roll': 0.0}}, {'location': {'x': -221.17080688476562, 'y': -22.212583541870117, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.90692138671875, 'roll': 0.0}}, {'location': {'x': -241.1834716796875, 'y': -22.04541778564453, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 179.48190307617188, 'roll': 0.0}}, {'location': {'x': -261.2293395996094, 'y': -21.813844680786133, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 177.923095703125, 'roll': 0.0}}, {'location': {'x': -281.347412109375, 'y': -19.575939178466797, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 169.38211059570312, 'roll': 0.0}}, {'location': {'x': -300.909912109375, 'y': -14.37500286102295, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 160.84112548828125, 'roll': 0.0}}, {'location': {'x': -319.4831237792969, 'y': -6.326372146606445, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 152.30014038085938, 'roll': 0.0}}, {'location': {'x': -336.64208984375, 'y': 4.3494462966918945, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 144.90008544921875, 'roll': 0.0}}, {'location': {'x': -352.09234619140625, 'y': 17.594667434692383, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 131.5033416748047, 'roll': 0.0}}, {'location': {'x': -363.31475830078125, 'y': 34.66670608520508, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 115.13504028320312, 'roll': 0.0}}, {'location': {'x': -369.2712707519531, 'y': 54.20941162109375, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 98.7667465209961, 'roll': 0.0}}, {'location': {'x': -370.3137512207031, 'y': 74.39646911621094, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 91.17143249511719, 'roll': 0.0}}, {'location': {'x': -370.7059326171875, 'y': 94.40699005126953, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 450.699951171875, 'roll': 0.0}}, {'location': {'x': -370.5911865234375, 'y': 114.45342254638672, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.14167785644531, 'roll': 0.0}}, {'location': {'x': -370.2915954589844, 'y': 134.451171875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.14167785644531, 'roll': 0.0}}, {'location': {'x': -369.99200439453125, 'y': 154.44894409179688, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.14167785644531, 'roll': 0.0}}, {'location': {'x': -369.6923828125, 'y': 174.44668579101562, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.14167785644531, 'roll': 0.0}}, {'location': {'x': -369.3927917480469, 'y': 194.44444274902344, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.14167785644531, 'roll': 0.0}}, {'location': {'x': -367.3905334472656, 'y': 214.74183654785156, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 433.3907470703125, 'roll': 0.0}}, {'location': {'x': -357.1452331542969, 'y': 232.638916015625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 407.0306701660156, 'roll': 0.0}}, {'location': {'x': -340.0597839355469, 'y': 244.1220245361328, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 21.97093963623047, 'roll': 0.0}}, {'location': {'x': -317.7245788574219, 'y': 249.9932403564453, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 7.152055263519287, 'roll': 0.0}}, {'location': {'x': -296.2068786621094, 'y': 250.78277587890625, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -276.2069091796875, 'y': 250.74679565429688, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -256.2069396972656, 'y': 250.7108154296875, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -236.2069854736328, 'y': 250.67481994628906, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -216.20703125, 'y': 250.6388397216797, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -196.20706176757812, 'y': 250.6028594970703, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -176.20709228515625, 'y': 250.56687927246094, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -156.20712280273438, 'y': 250.53089904785156, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.10308279097080231, 'roll': 0.0}}, {'location': {'x': -136.18673706054688, 'y': 250.46226501464844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -116.18685150146484, 'y': 250.392822265625, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -96.18697357177734, 'y': 250.32339477539062, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -76.18709564208984, 'y': 250.25396728515625, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -56.18722152709961, 'y': 250.1845245361328, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -36.187339782714844, 'y': 250.11508178710938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -16.18745994567871, 'y': 250.04563903808594, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -0.19892166554927826, 'roll': 0.0}}, {'location': {'x': -9.441421508789062, 'y': 262.8674011230469, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -270.9037780761719, 'roll': 0.0}}, {'location': {'x': -9.125947952270508, 'y': 282.86492919921875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 89.09619903564453, 'roll': 0.0}}, {'location': {'x': -8.904545783996582, 'y': 302.68487548828125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.20118713378906, 'roll': 0.0}}, {'location': {'x': -9.102336883544922, 'y': 322.6189880371094, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.60321044921875, 'roll': 0.0}}, {'location': {'x': -9.312891960144043, 'y': 342.61785888671875, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.60321044921875, 'roll': 0.0}}, {'location': {'x': -10.316596031188965, 'y': 362.4660949707031, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 94.32821655273438, 'roll': 0.0}}, {'location': {'x': -14.411600112915039, 'y': 381.4922790527344, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 109.96467590332031, 'roll': 0.0}}, {'location': {'x': -16.25929832458496, 'y': 401.9967346191406, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 65.10665130615234, 'roll': 0.0}}, {'location': {'x': 1.3293812274932861, 'y': 413.20587158203125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -1.8416810035705566, 'roll': 0.0}}, {'location': {'x': 17.961387634277344, 'y': 400.71075439453125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -69.39497375488281, 'roll': 0.0}}, {'location': {'x': 16.39168930053711, 'y': 380.1836853027344, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 256.85906982421875, 'roll': 0.0}}, {'location': {'x': 13.358766555786133, 'y': 360.6607666015625, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -95.71395874023438, 'roll': 0.0}}, {'location': {'x': 12.706541061401367, 'y': 340.8878479003906, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.39678955078125, 'roll': 0.0}}, {'location': {'x': 12.917095184326172, 'y': 320.8889465332031, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.39678955078125, 'roll': 0.0}}, {'location': {'x': 13.098448753356934, 'y': 300.7503662109375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.02361297607422, 'roll': 0.0}}, {'location': {'x': 12.840372085571289, 'y': 280.5563659667969, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.90380096435547, 'roll': 0.0}}, {'location': {'x': 12.524896621704102, 'y': 260.5588073730469, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -90.90380096435547, 'roll': 0.0}}, {'location': {'x': 23.45745086669922, 'y': 251.5336151123047, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 43.45744705200195, 'y': 251.54043579101562, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 63.45744323730469, 'y': 251.54725646972656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 83.45744323730469, 'y': 251.55409240722656, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 103.45744323730469, 'y': 251.5609130859375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 123.45744323730469, 'y': 251.56773376464844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 143.45745849609375, 'y': 251.57455444335938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}, {'location': {'x': 163.4574432373047, 'y': 251.5813751220703, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.019545903429389, 'roll': 0.0}}],
    'Carla/Maps/Town10HD': [{'location': {'x': 86.12206268310547, 'y': 135.372802734375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -30.75467300415039, 'roll': 0.0}}, {'location': {'x': 101.69976806640625, 'y': 119.30815887451172, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -59.33835983276367, 'roll': 0.0}}, {'location': {'x': 108.82305908203125, 'y': 98.4915771484375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -82.88043212890625, 'roll': 0.0}}, {'location': {'x': 109.33499145507812, 'y': 77.892578125, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -449.6092224121094, 'roll': 0.0}}, {'location': {'x': 109.47138977050781, 'y': 57.893043518066406, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -449.6092224121094, 'roll': 0.0}}, {'location': {'x': 109.6077880859375, 'y': 37.89350891113281, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.6092529296875, 'roll': 0.0}}, {'location': {'x': 109.74417877197266, 'y': 17.893972396850586, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.6092529296875, 'roll': 0.0}}, {'location': {'x': 109.88057708740234, 'y': -2.1055612564086914, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -89.6092529296875, 'roll': 0.0}}, {'location': {'x': 109.9718017578125, 'y': -22.06999969482422, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 270.779296875, 'roll': 0.0}}, {'location': {'x': 104.95696258544922, 'y': -43.847347259521484, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -117.44700622558594, 'roll': 0.0}}, {'location': {'x': 89.95840454101562, 'y': -60.514007568359375, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -146.5219268798828, 'roll': 0.0}}, {'location': {'x': 68.75068664550781, 'y': -67.79183959960938, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -175.59683227539062, 'roll': 0.0}}, {'location': {'x': 48.34845733642578, 'y': -67.9167251586914, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.97657775878906, 'roll': 0.0}}, {'location': {'x': 28.34845542907715, 'y': -67.90855407714844, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.97657775878906, 'roll': 0.0}}, {'location': {'x': 8.34846019744873, 'y': -67.90038299560547, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 179.97657775878906, 'roll': 0.0}}, {'location': {'x': -11.594415664672852, 'y': -68.11563873291016, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 180.59674072265625, 'roll': 0.0}}, {'location': {'x': -31.593332290649414, 'y': -68.32392883300781, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -179.40325927734375, 'roll': 0.0}}, {'location': {'x': -51.592247009277344, 'y': -68.5322265625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -179.40325927734375, 'roll': 0.0}}, {'location': {'x': -71.76483917236328, 'y': -68.72042846679688, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 178.70098876953125, 'roll': 0.0}}, {'location': {'x': -93.38449096679688, 'y': -61.789215087890625, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 145.74755859375, 'roll': 0.0}}, {'location': {'x': -107.92415618896484, 'y': -44.75082778930664, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 116.98052978515625, 'roll': 0.0}}, {'location': {'x': -113.46452331542969, 'y': -23.34680938720703, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 92.04415893554688, 'roll': 0.0}}, {'location': {'x': -113.7054672241211, 'y': -3.2199134826660156, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.6422348022461, 'roll': 0.0}}, {'location': {'x': -113.92964935302734, 'y': 16.778831481933594, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.6422348022461, 'roll': 0.0}}, {'location': {'x': -114.15382385253906, 'y': 36.7775764465332, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.6422348022461, 'roll': 0.0}}, {'location': {'x': -114.37799835205078, 'y': 56.77631759643555, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 90.6422348022461, 'roll': 0.0}}, {'location': {'x': -114.59477233886719, 'y': 76.8521957397461, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 449.80084228515625, 'roll': 0.0}}, {'location': {'x': -111.5401382446289, 'y': 98.0307846069336, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 433.7845764160156, 'roll': 0.0}}, {'location': {'x': -102.76068115234375, 'y': 117.54448699951172, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 417.76824951171875, 'roll': 0.0}}, {'location': {'x': -87.12371063232422, 'y': 133.28421020507812, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 31.67333984375, 'roll': 0.0}}, {'location': {'x': -65.93357849121094, 'y': 140.1871795654297, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 4.414215087890625, 'roll': 0.0}}, {'location': {'x': -45.56529235839844, 'y': 140.43115234375, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': -359.64788818359375, 'roll': 0.0}}, {'location': {'x': -25.565670013427734, 'y': 140.5540771484375, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.3521270751953125, 'roll': 0.0}}, {'location': {'x': -5.566047668457031, 'y': 140.677001953125, 'z': 1.0}, 'rotation': {'pitch': 360.0, 'yaw': 0.3521270751953125, 'roll': 0.0}}, {'location': {'x': 14.39871883392334, 'y': 140.9202117919922, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.7171252965927124, 'roll': 0.0}}, {'location': {'x': 34.43466567993164, 'y': 141.0443878173828, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.32044845819473267, 'roll': 0.0}}, {'location': {'x': 54.43435287475586, 'y': 141.15625, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': 0.32044845819473267, 'roll': 0.0}}, {'location': {'x': 75.72279357910156, 'y': 139.77456665039062, 'z': 1.0}, 'rotation': {'pitch': 0.0, 'yaw': -15.128620147705078, 'roll': 0.0}}]
}

class LaneDetector():

    def __init__(self, car, map, world, x_row, camera_transform, fov, n_points):
        self.inference_distances = {
            "Carla/Maps/Town10HD": 1000,
            "Carla/Maps/Town05": 1000,
            "Carla/Maps/Town02_Opt": 1000,
            "Carla/Maps/Town01": 1000,
            "Carla/Maps/Town01_Opt": 1000,
            "Carla/Maps/Town04": 2000,
            "Carla/Maps/Town06": 2000
        } # OJO que no los estás usando ahora. Los buenos están en controllerCarla
        self.n_points = n_points
        self.world = world
        self.last_valid_centers = None
        self.lane_points = None
        self.NON_DETECTED = -1
        self.NON_DETECTED = -1
        self.detection_mode = "carla_perfect"
        self.sync_mode = True
        self.show_images = False
        self.car = car
        self.map = map
        self.x_row = x_row
        self.no_detected = [[0]] * len(x_row)
        self.fov = fov

        if self.detection_mode == 'yolop':
            from utils.yolop.YOLOP import get_net
            import torchvision.transforms as transforms
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            # INIT YOLOP
            self.yolop_model = get_net()
            checkpoint = torch.load("/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/utils/yolop/weights/End-to-end.pth")
            self.yolop_model.load_state_dict(checkpoint['state_dict'])
        elif self.detection_mode == "lane_detector_v2":
            self.lane_model = torch.load(
                '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/models/CARLA/fastai_torch_lane_detector_model.pth')
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector_v2_poly":
            self.lane_model = torch.load(
                '/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/models/CARLA/fastai_torch_lane_detector_model.pth')
            self.lane_model.eval()
        elif self.detection_mode == "lane_detector":
            self.lane_model = torch.load('/home/ruben/Desktop/my-BehaviorMetrics/behavior_metrics/models/CARLA/best_model_torch.pth')
            self.lane_model.eval()
        else:
            self.trafo_matrix_vehicle_to_cam = np.array(
                camera_transform.get_inverse_matrix()
            )
            self.k = None



    def choose_lane(self, distance_to_center_normalized, center_points):
        close_lane_indexes = [min(enumerate(inner_array), key=lambda x: abs(x[1]))[0] for inner_array in
                              distance_to_center_normalized]
        distances = [array[index] for array, index in zip(distance_to_center_normalized, close_lane_indexes)]
        centers = [array[index] for array, index in zip(center_points, close_lane_indexes)]
        return distances, centers


    def get_resized_image(self, sensor_data, new_width=640):
        # Assuming sensor_data is the image obtained from the sensor
        # Convert sensor_data to a numpy array or PIL Image if needed
        # For example, if sensor_data is a numpy array:
        # sensor_data = Image.fromarray(sensor_data)
        sensor_data = np.array(sensor_data, copy=True)

        # Get the current width and height
        height = sensor_data.shape[0]
        width = sensor_data.shape[1]

        # Calculate the new height to maintain the aspect ratio
        new_height = int((new_width / width) * height)

        resized_img = Image.fromarray(sensor_data).resize((new_width, new_height))

        # Convert back to numpy array if needed
        # For example, if you want to return a numpy array:
        resized_img_np = np.array(resized_img)

        return resized_img_np


    def detect_lane_detector(self, raw_image):
        image_tensor = raw_image.transpose(2, 0, 1).astype('float32') / 255
        x_tensor = torch.from_numpy(image_tensor).to("cuda").unsqueeze(0)
        model_output = torch.softmax(self.lane_model.forward(x_tensor), dim=1).cpu().numpy()
        return model_output


    def detect_yolop(self, raw_image):
        # Run inference
        img = self.transform(raw_image)
        img = img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        det_out, da_seg_out, ll_seg_out = self.yolop_model(img)

        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, scale_factor=int(1), mode='bicubic')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        return ll_seg_mask


    def lane_detection_overlay(self, image, left_mask, right_mask):
        res = np.copy(image)
        # We show only points with probability higher than 0.5
        res[left_mask > 0.5, :] = [255, 0, 0]
        res[right_mask > 0.5, :] = [0, 0, 255]
        return res

    def set_inait_pose(self):
        import random

        spawn_dict = random.choice(POINTS_PER_MAP[self.map.name])
        location = carla.Location(
            x=spawn_dict['location']['x'],
            y=spawn_dict['location']['y'],
            z=spawn_dict['location']['z']
        )
        yaw_offset = random.uniform(-4.0, 4.0)
        if random.random() < 0.5:
            yaw_offset += 180.0  # Flip direction
        new_rotation = carla.Rotation(
            pitch=spawn_dict['rotation']['pitch'],
            yaw=spawn_dict['rotation']['yaw'] + yaw_offset,
            roll=spawn_dict['rotation']['roll']
        )

        new_transform = carla.Transform(location, new_rotation)
        # new_transform = carla.Transform(carla.Location(x=381.017059, y=251.655594, z=1.000000),
        #           carla.Rotation(pitch=0.000000, yaw=-1.589903, roll=0.000000))
        # new_transform = carla.Transform(spawn_dict)


        print(new_transform)
        self.car.set_transform(new_transform)

    def post_process(self, ll_segment):
        ''''
        Lane line post-processing
        '''
        # ll_segment = morphological_process(ll_segment, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # ll_segment = morphological_process(ll_segment, kernel_size=20, func_type=cv2.MORPH_CLOSE)
        # return ll_segment
        # ll_segment = morphological_process(ll_segment, kernel_size=4, func_type=cv2.MORPH_OPEN)
        # ll_segment = morphological_process(ll_segment, kernel_size=8, func_type=cv2.MORPH_CLOSE)

        # Step 1: Create a binary mask image representing the trapeze
        mask = np.zeros_like(ll_segment)
        # pts = np.array([[300, 250], [-500, 600], [800, 600], [450, 260]], np.int32)
        pts = np.array([[280, 100], [-150, 600], [730, 600], [440, 100]], np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))  # Fill trapeze region with white (255)

        # Step 2: Apply the mask to the original image
        ll_segment_masked = cv2.bitwise_and(ll_segment, mask)
        # Apply the exclusion mask to ll_segment
        ll_segment_excluding_mask = cv2.bitwise_not(mask)
        ll_segment_excluded = cv2.bitwise_and(ll_segment, ll_segment_excluding_mask)
        self.display_image(ll_segment_excluded) if self.show_images else None
        self.display_image(ll_segment_masked) if self.show_images else None
        self.display_image(mask) if self.show_images else None

        return ll_segment_masked


    def detect_lines(self, raw_image):
        # if self.detection_mode == 'programmatic':
        #     gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        #     # mask_white = cv2.inRange(gray, 200, 255)
        #     # mask_image = cv2.bitWiseAnd(gray, mask_white)
        #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
        #     ll_segment = cv2.Canny(blur, 50, 100)
        #     cv2.imshow("raw", ll_segment)
        #     processed = self.post_process(ll_segment)
        #     lines = self.post_process_hough_programmatic(processed)
        if self.detection_mode == 'yolop':
            with torch.no_grad():
                ll_segment = (self.detect_yolop(raw_image) * 255).astype(np.uint8)
            # processed = self.post_process(ll_segment)
            lines = self.post_process_hough_yolop(ll_segment)
        elif self.detection_mode == 'lane_detector_v2':
            with torch.no_grad():
                ll_segment, left_mask, right_mask = self.detect_lane_detector(raw_image)[0]
            ll_segment = np.zeros_like(raw_image)
            ll_segment = self.lane_detection_overlay(ll_segment, left_mask, right_mask)
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            # Extract blue and red channels
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            lines = []
            left_line = self.post_process_hough_lane_det(blue_channel)
            if left_line is not None:
                lines.append([left_line])
            right_line = self.post_process_hough_lane_det(red_channel)
            if right_line is not None:
                lines.append([right_line])
            ll_segment = 0.5 * blue_channel + 0.5 * red_channel
            ll_segment = cv2.convertScaleAbs(ll_segment)
        elif self.detection_mode == 'lane_detector_v2_poly':
            with torch.no_grad():
                ll_segment, left_mask, right_mask = self.detect_lane_detector(raw_image)[0]
            ll_segment = np.zeros_like(raw_image)
            ll_segment = self.lane_detection_overlay(ll_segment, left_mask, right_mask)
            cv2.imshow("raw", ll_segment) if self.sync_mode and self.show_images else None
            # Extract blue and red channels
            blue_channel = ll_segment[:, :, 0]  # Blue channel
            red_channel = ll_segment[:, :, 2]  # Red channel

            ll_segment_left = self.post_process_hough_lane_det_poly(blue_channel)
            ll_segment_right = self.post_process_hough_lane_det_poly(red_channel)
            ll_segment = 0.5 * ll_segment_left if ll_segment_left is not None else np.zeros_like(blue_channel)
            ll_segment = ll_segment + 0.5 * ll_segment_right if ll_segment_right is not None else ll_segment
            ll_segment = cv2.convertScaleAbs(ll_segment)
            return ll_segment.astype(np.uint8), False
        elif self.detection_mode == 'carla_perfect':
            ll_segment = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

            heigth = ll_segment.shape[0] # 512
            width = ll_segment.shape[1] # 640

            trafo_matrix_global_to_camera = get_matrix_global(self.car, self.trafo_matrix_vehicle_to_cam)
            if self.k is None:
                self.k = get_intrinsic_matrix(self.fov, width, heigth)

            waypoint = self.map.get_waypoint(
                self.car.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            _, center_distance, alignment = self.get_lane_position(self.car, self.map)
            # print(f"dist {center_distance}")
            opposite = alignment < 0.5
            misalignment = (1 - abs(alignment)) * 10

            center_list, left_boundary, right_boundary, type_lane = create_lane_lines(waypoint, self.car, opposite=opposite)

            projected_left_boundary = project_polyline(
                left_boundary, trafo_matrix_global_to_camera, self.k, ll_segment.shape).astype(np.int32)
            projected_right_boundary = project_polyline(
                right_boundary, trafo_matrix_global_to_camera, self.k, ll_segment.shape).astype(np.int32)

            if (not check_inside_image(projected_right_boundary, width, heigth)
                    or not check_inside_image(projected_right_boundary, width, heigth)):
                return ll_segment, misalignment, center_distance, np.empty((0, 2)), np.empty((0, 2))
            image = np.zeros_like(ll_segment, dtype=np.uint8)
            self.draw_line_through_points(projected_left_boundary, image)
            self.draw_line_through_points(projected_right_boundary, image)
            return image, misalignment, center_distance, projected_left_boundary, projected_right_boundary

        detected_lines = self.merge_and_extend_lines(lines, ll_segment)

        # line_mask = morphological_process(line_mask, kernel_size=15, func_type=cv2.MORPH_CLOSE)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = cv2.dilate(line_mask, (15, 15), iterations=15)
        # line_mask = cv2.erode(line_mask, (5, 5), iterations=20)

        # TODO (Ruben) It is quite hardcoded and unrobust. Fix this to enable all lines and more than

        # 1 lane detection and cameras in other positions
        boundary_y = ll_segment.shape[1] * 2 // 5
        # Copy the lower part of the source image into the target image
        ll_segment[boundary_y:, :] = detected_lines[boundary_y:, :]
        ll_segment = (ll_segment // 255).astype(np.uint8)  # Keep the lower one-third of the image

        return ll_segment

    def calculate_curvature_from(self, right_lane_normalized_distances):
        x = np.array(self.x_row)
        y = np.array(right_lane_normalized_distances)

        coefficients = np.polyfit(x, y, 2)  # Returns [a, b, c]
        a, b, c = coefficients

        x_mid = x[2]  # Use the middle point
        y_prime = 2 * a * x_mid + b
        y_double_prime = 2 * a
        curvature = abs(y_double_prime) / ((1 + y_prime ** 2) ** (3 / 2))
        return curvature

    def calculate_max_curveture_from_centers(self, center_points):
        curvatures = []
        for i in range(1, len(center_points) - 1):
            k = curvature_from_three_points(
                np.array(center_points[i - 1]),
                np.array(center_points[i]),
                np.array(center_points[i + 1])
            )
            curvatures.append(k)
        return max(curvatures)

    def get_lane_position(self, vehicle: carla.Vehicle, map: carla.Map):
        """
        Determines the vehicle's position relative to the lane.

        Returns:
            A dictionary containing:
            - lane_side: "left", "right", or "center"
            - lane_offset: Distance from the lane center (positive: left, negative: right)
            - lane_alignment: Alignment of vehicle and lane directions (1: aligned, -1: opposite)
        """

        waypoint = map.get_waypoint(
            vehicle.get_transform().location, project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # Vehicle's forward vector
        vehicle_forward = vehicle.get_transform().get_forward_vector()
        vehicle_forward_np = np.array([vehicle_forward.x, vehicle_forward.y])

        # Lane's forward vector
        waypoint_forward = waypoint.transform.get_forward_vector()
        waypoint_forward_np = np.array([waypoint_forward.x, waypoint_forward.y])

        # Vector from waypoint to vehicle
        vehicle_location = vehicle.get_transform().location
        waypoint_location = waypoint.transform.location
        waypoint_to_vehicle = carla.Location(
            vehicle_location.x - waypoint_location.x,
            vehicle_location.y - waypoint_location.y,
            vehicle_location.z - waypoint_location.z
        )
        waypoint_to_vehicle_np = np.array([waypoint_to_vehicle.x, waypoint_to_vehicle.y])

        # 1. Lane Side (Left/Right)
        cross_product = np.cross(vehicle_forward_np, waypoint_forward_np)
        lane_side = "center"
        if cross_product > 0.1:
            lane_side = "left"
        elif cross_product < -0.1:
            lane_side = "right"

        # 2. Lane Offset (Distance to Center)
        # Project waypoint_to_vehicle onto a vector perpendicular to lane_forward
        lane_right_np = np.array([-waypoint_forward_np[1], waypoint_forward_np[0]])  # 90-degree rotation
        lane_offset = np.dot(waypoint_to_vehicle_np, lane_right_np)
        lane_offset /= np.linalg.norm(lane_right_np)

        # 3. Lane Alignment (Forward/Backward)/home/ruben/Desktop/2020-phd-ruben-lucas/src/RL-Studio/rl_studio/checkpoints/follow_lane_carla_sac_auto_carla_baselines/20250506-082233/best_model.zip
        lane_alignment = np.dot(vehicle_forward_np, waypoint_forward_np)

        return lane_side, lane_offset, lane_alignment

    def move_car_to_lane_center(self, vehicle, world, carla_map):
        # 1. Get the current location
        current_location = vehicle.get_transform().location

        # 2. Find the closest driving lane waypoint (center of the lane)
        waypoint = carla_map.get_waypoint(
            current_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        # 3. Create a transform from the waypoint: it has both location and rotation (forward direction)
        lane_center_transform = waypoint.transform

        # 4. Move the car there (teleport it)
        vehicle.set_transform(lane_center_transform)

        # Optional: wait a tick to update simulation
        world.tick()

        print("[INFO] Vehicle moved to center of lane at:", lane_center_transform.location)

    def draw_line_through_points(self, points, image):
        # Convert the points list to a format compatible with OpenCV (a numpy array)
        points = np.array(points, dtype=np.int32)  # Make sure points are integers

        # Draw a polyline through the points
        cv2.polylines(image, [points], isClosed=False, color=(255, 0, 0), thickness=2)

        return image


    def display_image(self, ll_segment):
        # Display the image
        pil_image = Image.fromarray(ll_segment)

        # Display the image using PIL
        pil_image.show()


    def post_process_hough_yolop(self, ll_segment):
        # Step 4: Perform Hough transform to detect lines
        lines = cv2.HoughLinesP(
            ll_segment,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 60,  # Angle resolution in radians
            threshold=8,  # Min number of votes for valid line
            minLineLength=8,  # Min allowed length of line
            maxLineGap=20  # Max allowed gap between line for joining them
        )

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Draw the detected lines on the blank image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Draw lines in white (255, 255, 255)

        # Apply dilation to the line image

        edges = cv2.Canny(line_mask, 50, 100)

        # Reapply HoughLines on the dilated image
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 90,  # Angle resolution in radians
            threshold=35,  # Min number of votes for valid line
            minLineLength=15,  # Min allowed length of line
            maxLineGap=20  # Max allowed gap between line for joining them
        )
        # Sort lines by their length
        # lines = sorted(lines, key=lambda x: x[0][0] * np.sin(x[0][1]), reverse=True)[:5]

        # Create a blank image to draw lines
        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Iterate over points
        for points in lines if lines is not None else []:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(line_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Postprocess the detected lines
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_OPEN)
        # line_mask = morphological_process(line_mask, kernel_size=5, func_type=cv2.MORPH_CLOSE)
        # kernel = np.ones((3, 3), np.uint8)  # Adjust the size as needed
        # eroded_image = cv2.erode(line_mask, kernel, iterations=1)

        return lines


    def post_process_hough_lane_det(self, ll_segment):
        # ll_segment = cv2.dilate(ll_segment, (3, 3), iterations=4)
        # ll_segment = cv2.erode(ll_segment, (3, 3), iterations=2)
        cv2.imshow("preprocess", ll_segment) if self.show_images else None
        # edges = cv2.Canny(ll_segment, 50, 100)
        # Extract coordinates of non-zero points
        nonzero_points = np.argwhere(ll_segment == 255)
        if len(nonzero_points) == 0:
            return None

        # Extract x and y coordinates
        x = nonzero_points[:, 1].reshape(-1, 1)  # Reshape for scikit-learn input
        y = nonzero_points[:, 0]

        # Fit linear regression model
        model = LinearRegression()
        model.fit(x, y)

        # Predict y values based on x
        y_pred = model.predict(x)

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Draw the linear regression line
        for i in range(len(x)):
            cv2.circle(line_mask, (x[i][0], int(y_pred[i])), 2, (255, 0, 0), -1)

        cv2.imshow("result", line_mask) if self.show_images else None

        # Find the minimum and maximum x coordinates
        min_x = np.min(x)
        max_x = np.max(x)

        # Find the corresponding predicted y-values for the minimum and maximum x coordinates
        y1 = int(model.predict([[min_x]]))
        y2 = int(model.predict([[max_x]]))

        # Define the line segment
        line_segment = (min_x, y1, max_x, y2)

        return line_segment


    def merge_and_extend_lines(self, lines, ll_segment):
        # Merge parallel lines
        merged_lines = []
        for line in lines if lines is not None else []:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Compute the angle of the line

            # Check if there is a similar line in the merged lines
            found = False
            for merged_line in merged_lines:
                angle_diff = abs(merged_line['angle'] - angle)
                if angle_diff < 20 and abs(angle) > 25:  # Adjust this threshold based on your requirement
                    # Merge the lines by averaging their coordinates
                    merged_line['x1'] = (merged_line['x1'] + x1) // 2
                    merged_line['y1'] = (merged_line['y1'] + y1) // 2
                    merged_line['x2'] = (merged_line['x2'] + x2) // 2
                    merged_line['y2'] = (merged_line['y2'] + y2) // 2
                    found = True
                    break

            if not found and abs(angle) > 25:
                merged_lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'angle': angle})

        # Draw the merged lines on the original image
        merged_image = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8
        # if len(merged_lines) < 2 or len(merged_lines) > 2:
        #    print("ii")
        for line in merged_lines:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            cv2.line(merged_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Display the original image with merged lines
        # cv2.imshow('Merged Lines', merged_image) if self.sync_mode and self.show_images else None

        line_mask = np.zeros_like(ll_segment, dtype=np.uint8)  # Ensure dtype is uint8

        # Step 5: Perform linear regression on detected lines
        # Iterate over detected lines
        for line in merged_lines if lines is not None else []:
            # Extract endpoints of the line
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']

            # Fit a line to the detected points
            vx, vy, x0, y0 = cv2.fitLine(np.array([[x1, y1], [x2, y2]], dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate the slope and intercept of the line
            slope = vy / vx

            # Extend the line if needed (e.g., to cover the entire image width)
            extended_y1 = ll_segment.shape[0] - 1  # Bottom of the image
            extended_x1 = x0 + (extended_y1 - y0) / slope
            extended_y2 = 0  # Upper part of the image
            extended_x2 = x0 + (extended_y2 - y0) / slope

            if extended_x1 > 2147483647 or extended_x2 > 2147483647 or extended_y1 > 2147483647 or extended_y2 > 2147483647:
                cv2.line(line_mask, (int(x0), 0), (int(x0), ll_segment.shape[0] - 1), (255, 0, 0), 2)
                continue
            # Draw the extended line on the image
            cv2.line(line_mask, (int(extended_x1), extended_y1), (int(extended_x2), extended_y2), (255, 0, 0), 2)
        return line_mask


    def discard_not_confident_centers(self, center_lane_indexes):
        # Count the occurrences of each list size leaving out of the equation the non-detected
        size_counter = Counter(len(inner_list) for inner_list in center_lane_indexes)
        # Check if size_counter is empty, which mean no centers found
        if not size_counter:
            return center_lane_indexes
        # Find the most frequent size
        # most_frequent_size = max(size_counter, key=size_counter.get)

        # Iterate over inner lists and set elements to 1 if the size doesn't match majority
        result = []
        for inner_list in center_lane_indexes:
            # if len(inner_list) != most_frequent_size:
            if len(inner_list) < 1:  # If we don't see the 2 lanes, we discard the row
                inner_list = [0] * len(inner_list)  # Set all elements to 1
            result.append(inner_list)

        return result

    def calculate_center(self, image):
        width = image.shape[1]
        center_image = width / 2
        ## get total lines in every line point
        lines = [image[self.x_row[i], :] for i, _ in enumerate(self.x_row)]
        # ## As we drive in the right lane, we get from right to left
        # lines_inversed = [list(reversed(lines[x])) for x, _ in enumerate(lines)]
        ## get the distance from the right lane to center
        center_lane_indexes, lane_indexes = zip (*[
            self.find_lane_center(lines[x], x, center_image) for x, _ in enumerate(lines)
        ])

        # last_lane_indexes = self.find_first_lanes_index(image)

        if all(x in ([1], [-1]) for x in center_lane_indexes):
            center_lane_indexes = self.no_detected
        # this part consists of checking the number of lines detected in all rows
        # then discarding the rows (set to 1) in which more or less centers are detected
        # center_lane_indexes = self.discard_not_confident_centers(center_lane_indexes)

        center_lane_distances = [
            [center_image - x for x in inner_array] for inner_array in center_lane_indexes
        ]

        # Calculate the average position of the lane lines
        ## normalized distance
        distance_to_center_normalized = [
            np.array(x) / (width - center_image) for x in center_lane_distances
        ]
        return center_lane_indexes, distance_to_center_normalized


    def draw_dash(self, index, dist, ll_segment):
        height, width = ll_segment.shape

        # List of relative positions around 'dist' to set
        offsets = [-5, -4, -3, -2, -1]

        for offset in offsets:
            x = index
            y = dist + offset
            if 0 <= x < height and 0 <= y < width:
                ll_segment[x, y] = 255


    def add_midpoints(self, ll_segment, index, dist):
        # Set the value at the specified index and distance to 1
        self.draw_dash(index, dist, ll_segment)
        self.draw_dash(index + 2, dist, ll_segment)
        self.draw_dash(index + 1, dist, ll_segment)
        self.draw_dash(index - 1, dist, ll_segment)
        self.draw_dash(index - 2, dist, ll_segment)

    def show_ll_seg_image(self, centers, ll_segment, interpolated_left, interpolated_right, suffix="", name='ll_seg'):
        if self.detection_mode == "carla_perfect":
            ll_segment_int8 = ll_segment
        else:
            ll_segment_int8 = (ll_segment * 255).astype(np.uint8)

        height, width = ll_segment.shape
        blank_image = np.zeros((height, width), dtype=np.uint8)
        # ll_segment_all = [blank_image.copy(), blank_image.copy(), blank_image.copy()]

        ll_segment_all = [np.copy(ll_segment_int8), np.copy(ll_segment_int8), np.copy(ll_segment_int8)]

        # Draw the midpoint used as right center lane
        for index, dist in zip(self.x_row, centers):
            self.add_midpoints(ll_segment_all[0], index, dist)

        # Draw a line for the selected perception points
        for index in self.x_row:
            for i in range(630):
                ll_segment_all[0][index][i] = 255

        # Add interpolated lane points to the first channel
        for pt in interpolated_left:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < ll_segment_all[0].shape[0] and 0 <= x < ll_segment_all[0].shape[1]:
                self.add_midpoints(ll_segment_all[0], y, x)
        for pt in interpolated_right:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < ll_segment_all[0].shape[0] and 0 <= x < ll_segment_all[0].shape[1]:
                self.add_midpoints(ll_segment_all[0], y, x)

        ll_segment_stacked = np.stack(ll_segment_all, axis=-1)
        return ll_segment_stacked

    def find_lane_center(self, mask, i, center_image):
        # Find the indices of 1s in the array
        mask_array = np.array(mask)
        indices = np.where(mask_array > 0.8)[0]

        # If there are no 1s or only one set of 1s, return None
        if len(indices) < 2:
            # TODO (Ruben) For the moment we will be dealing with no detection as a fixed number
            return self.miss_detection(i, center_image), []

        # Find the indices where consecutive 1s change to 0
        diff_indices = np.where(np.diff(indices) > 1)[0]
        # If there is only one set of 1s, return None
        if len(diff_indices) == 0:
            return self.miss_detection(i, center_image), []

        interested_line_borders = np.array([], dtype=np.int8)

        for index in diff_indices:
            interested_line_borders = np.append(interested_line_borders, indices[index])
            interested_line_borders = np.append(interested_line_borders, int(indices[index + 1]))

        midpoints = self.calculate_midpoints(interested_line_borders)
        self.no_detected[i] = midpoints

        return midpoints, interested_line_borders


    def calculate_midpoints(self, input_array):
        midpoints = []
        for i in range(0, len(input_array) - 1, 2):
            midpoint = (input_array[i] + input_array[i + 1]) // 2
            midpoints.append(midpoint)
        return midpoints

    def get_stable_lane_lines(self, segment_length: int = 80,
                              opposite: bool = False,
                              exclude_junctions: bool = False,
                              only_turns: bool = False):
        """
        Returns a stable rolling lane poly-line.
        On the first call it builds the N-point list from the car's current waypoint.
        On subsequent calls it only advances the list when the car has really moved
        past the first stored point, preventing abrupt lane switches.

        Returns
        -------
        center_list : np.ndarray  shape=(N,3)
        left_boundary : np.ndarray
        right_boundary : np.ndarray
        current_wp : carla.Waypoint       (the new reference waypoint)
        """

        # ------------------------------------------------------------------
        # 1)  Build the poly-line once
        # ------------------------------------------------------------------
        if self.lane_points is None:
            wp = self.map.get_waypoint(
                self.car.get_transform().location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            center, left_b, right_b, last_wp = create_lane_lines(
                wp, opposite=opposite,
                exclude_junctions=exclude_junctions,
                only_turns=only_turns)

            # Keep both numpy arrays *and* the last carla.Waypoint for fast extension
            self.lane_points = {
                "center": center.tolist(),  # lists are easier to pop/append
                "last_wp": last_wp  # last waypoint object
            }

        while len(self.lane_points["center"]) < 90:
            last_wp = self.lane_points["last_wp"]

            # Try to move forward or backward depending on direction
            if opposite:
                next_wps = last_wp.previous(1.0)
            else:
                next_wps = last_wp.next(1.0)

            if not next_wps:
                break  # No more waypoints available, cannot extend further

            # Get new waypoint
            new_wp = next_wps[0]
            self.lane_points["last_wp"] = new_wp  # update stored last_wp

            # Compute new center/left/right lane points
            center_np = carla_vec_to_np_array(new_wp.transform.location)

            self.lane_points["center"].append(center_np.tolist())

        # ------------------------------------------------------------------
        # 3)  Return as numpy arrays
        # ------------------------------------------------------------------
        center_arr = np.asarray(self.lane_points["center"])
        return center_arr, self.lane_points["last_wp"]

    def process_image(self, image):
        # raw_image = self.get_resized_image(image)
        raw_image = image

        (ll_segment,
         misalignment,
         center_distance,
         center_points) = self.detect_center_line_perfect(image, n_points=self.n_points)

        centers_image = np.zeros(raw_image.shape, dtype=np.uint8)
        for index in range(len(center_points)):
            cv2.circle(centers_image, (center_points[index][0], center_points[index][1]), radius=3,
                       color=(255, 255, 0), thickness=-1)
            # cv2.circle(ll_segment, (line_points[index][0], line_points[index][1]), radius=3, color=(0, 0, 255),
            #            thickness=-1)
        gray_overlay = cv2.cvtColor(centers_image, cv2.COLOR_BGR2GRAY)
        mask = gray_overlay > 10
        mask_3ch = np.stack([mask] * 3, axis=-1)

        stacked_image = np.where(mask_3ch, centers_image, raw_image)
        # self.add_text_to_image(stacked_image)

        return center_points, stacked_image, center_distance, misalignment

    def miss_detection(self, i, center_image):
        return [int((center_image * 2) - 1)] if self.no_detected[i][0] > center_image else [0]

    def find_first_lanes_index(self, image):
        height, width = image.shape
        for i in range(height):
            line = image[i, :]
            indices = np.where(line > 0.8)[0]

            if len(indices) < 2:
                continue

            # Find gaps to split continuous segments
            gaps = np.where(np.diff(indices) > 1)[0]
            splits = np.split(indices, gaps + 1)

            interested_line_borders = []

            for segment in splits:
                if len(segment) >= 2:
                    # Save start and end of each segment
                    interested_line_borders.append([i, segment[0]])
                    interested_line_borders.append([i, segment[-1]])

            if interested_line_borders:
                return np.array(interested_line_borders, dtype=np.int16)

        return None

    def detect_center_line_perfect(self, ll_segment, n_points=20):
        ll_segment = cv2.cvtColor(ll_segment, cv2.COLOR_BGR2GRAY)

        height = ll_segment.shape[0]
        width = ll_segment.shape[1]

        trafo_matrix_global_to_camera = get_matrix_global(self.car, self.trafo_matrix_vehicle_to_cam)

        if self.k is None:
            self.k = get_intrinsic_matrix(90, width, height)

        _, center_distance, alignment = self.get_lane_position(self.car, self.map)
        opposite = alignment < 0.5
        misalignment = (1 - abs(alignment)) * 10

        center_list, current_wp = self.get_stable_lane_lines(opposite=opposite)

        if center_list is None or len(center_list) < 2:
            interpolated_center = np.full((n_points, 2), self.NON_DETECTED)
            self.lane_points = None
        else:
            projected_center = project_polyline(
                center_list, trafo_matrix_global_to_camera, self.k, image_shape=ll_segment.shape
            ).astype(np.int32)

            # Discard non visible points from projected centers
            h, w = ll_segment.shape[:2]
            mask = np.array([
                0 <= pt[0] < w and 0 <= pt[1] < h
                for pt in projected_center
            ])

            if np.sum(mask) < 2:
                # if self.last_valid_centers is not None:
                #     interpolated_center = self.last_valid_centers
                # else:
                interpolated_center = np.full((n_points, 2), self.NON_DETECTED)
            else:
                # Apply the same mask to both 2D and 3D data
                visible_center = projected_center[mask]
                # In lane_points just remove the points that are not visible because they are behind the car
                for i, keep in enumerate(mask):
                    if keep:
                        first_true_index = i
                        break
                else:
                    # If mask is all False, remove all points
                    first_true_index = len(mask)
                self.lane_points["center"] = self.lane_points["center"][first_true_index:]

                interpolated_center = interpolate_lane_points(visible_center, n_points)
                self.last_valid_centers = interpolated_center
        return ll_segment, misalignment, center_distance, interpolated_center

    def add_text_to_image(self, image):
        waypoint = self.map.get_waypoint(
            self.car.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        road_id_text = f"Road ID: {waypoint.road_id}"
        lane_id_text = f"Lane ID: {waypoint.lane_id}"
        section_id_text = f"Section ID: {waypoint.section_id}"

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White color (BGR) for the text
        bg_color = (0, 0, 0)  # Black color (BGR) for background rectangle (optional)
        line_height = 30  # Pixels per line

        # Starting position for the text (top-left corner of the image)
        # Adjust x and y coordinates as needed
        x_start = 10
        y_start = 30

        # It's good practice to add a semi-transparent background for readability
        # This example just puts text directly. For background, you'd draw a filled rectangle first.

        # Draw each line of text
        cv2.putText(image, road_id_text, (x_start, y_start),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(image, lane_id_text, (x_start, y_start + line_height),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(image, section_id_text, (x_start, y_start + 2 * line_height),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def calculate_v_goal(self, mean_curvature, center_distance, deviated_points):
        dist_error = abs(center_distance) * 10
        close_error = 0

        mean_curv = max(0, mean_curvature - 1) * 10

        if deviated_points >= self.n_points / 2:
            mean_curv = max(0, mean_curvature - 1) * 30
            close_error = 9

        elif deviated_points >= self.n_points / 3:
            mean_curv = max(0, mean_curvature - 1) * 20
            close_error = 6

        elif deviated_points >= self.n_points / 4:
            mean_curv = max(0, mean_curvature - 1) * 10
            close_error = 3

        v_goal = max(9, 25 - (mean_curv + dist_error))
        v_goal = max(2, v_goal - (close_error))

        return v_goal


    # def calculate_v_goal(self, mean_curvature, center_distance, y_normalized):
    #     mean_curv = max(0, mean_curvature - 1) * 25
    #     dist_error = abs(center_distance) * 10
    #     v_goal = max(7, 25 - (mean_curv + dist_error))
    #
    #     farther_y = y_normalized[-1]
    #     if farther_y > 0.65:
    #         v_goal = max(3, v_goal - 4)
    #
    #     # print(f"monitoring last modification bro! dist_minus -> {dist_error}")
    #     # print(f"monitoring last modification bro! curv_minus -> {curv}")
    #     # print(f"monitoring last modification bro! v_goal -> {v_goal}")
    #     return  v_goal

    def average_curvature_from_centers(self, center_points):
        total_angle = 0.0

        for i in range(1, len(center_points) - 1):
            p1 = np.array(center_points[i - 1])
            p2 = np.array(center_points[i])
            p3 = np.array(center_points[i + 1])

            v1 = p2 - p1
            v2 = p3 - p2

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                continue  # skip degenerate segment

            cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
            angle = np.arccos(cos_theta)  # radians

            total_angle += angle

        return total_angle  # in radians

    def normalize_centers(self, centers):
        x_centers = centers[:, 0]
        x_centers_normalized = (x_centers / 640).tolist()
        states = x_centers_normalized
        y_centers = centers[:, 1]
        y_centers_normalized = (y_centers / 512).tolist()
        states = states + y_centers_normalized  # Returns a list
        return states, x_centers_normalized, y_centers_normalized


    # def normalize_centers(self, centers):
    #     x_centers = centers[:, 0]
    #     x_centers_normalized = (x_centers / 512).tolist()
    #     states = x_centers_normalized
    #     y_centers = centers[:, 1]
    #     y_centers_normalized = (y_centers / 640).tolist()
    #     states = states + y_centers_normalized  # Returns a list
    #     return states, x_centers_normalized, y_centers_normalized

# def interpolate_lane_points(lane_points: np.ndarray, num_points: int = 10) -> np.ndarray:
#     """
#     Interpolates `num_points` equidistant points along a polyline defined by `lane_points`.
#
#     Args:
#         lane_points (np.ndarray): Array of shape (N, 2) representing the lane boundary points.
#         num_points (int): Number of interpolated points to return.
#
#     Returns:
#         np.ndarray: Interpolated points along the lane, shape (num_points, 2)
#     """
#     if lane_points.shape[0] < 2:
#         raise ValueError("Need at least two points to interpolate a lane.")
#
#     # Compute cumulative arc length (distance) along the lane
#     deltas = np.diff(lane_points, axis=0)
#     segment_lengths = np.linalg.norm(deltas, axis=1)
#     cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)
#
#     # Target arc lengths for evenly spaced points
#     target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
#
#     # Interpolated x and y separately over arc length
#     interp_x = np.interp(target_lengths, cumulative_lengths, lane_points[:, 0])
#     interp_y = np.interp(target_lengths, cumulative_lengths, lane_points[:, 1])
#
#     return np.stack((interp_x, interp_y), axis=1).astype(np.int32)

def curvature_from_three_points(p1, p2, p3):
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
    if area == 0:
        return 0
    return (4 * area) / (a * b * c)

def intersect_with_image_border(p1, p2, w, h):
    from shapely.geometry import LineString, box

    line = LineString([p1, p2])
    image_box = box(0, 0, w - 1, h - 1)
    inter = line.intersection(image_box)

    if inter.is_empty:
        return None
    if inter.geom_type == "Point":
        return np.array(inter.coords[0])
    elif inter.geom_type == "MultiPoint":
        return np.array(inter.geoms[0].coords[0])  # Pick one
    return None

def normalize(value, maximum):
    return ((value / maximum) * 2) - 1

def carla_vec_to_np_array(vec):
    """
    Converts a CARLA Location or Vector3D into a NumPy array [x, y, z].
    """
    return np.array([vec.x, vec.y, vec.z])

def interpolate_lane_points(lane_points: np.ndarray, num_points: int = 20, start_y: int = 640) -> np.ndarray:
    """
    Interpolates `num_points` equidistant points along a polyline that starts at y = `start_y`
    (image bottom) and ends at lane_points[-1], extrapolating the initial segment if needed.

    Args:
        lane_points (np.ndarray): Array of shape (N, 2) representing the lane boundary points.
        num_points (int): Number of interpolated points to return.
        start_y (int): The y-coordinate from which to start the interpolation (e.g., 640).

    Returns:
        np.ndarray: Interpolated points along the extended lane, shape (num_points, 2)
    """
    if lane_points.shape[0] < 2:
        return np.zeros((num_points, 2), dtype=np.float32)

    p0 = lane_points[0]
    p1 = lane_points[1]

    dy = p1[1] - p0[1]
    dx = p1[0] - p0[0]

    if dy == 0:
        slope = 0
    else:
        slope = dx / dy

    # Extrapolate backward to y = start_y
    delta_y = start_y - p0[1]
    extrapolated_x = p0[0] + slope * delta_y
    extrapolated_point = np.array([extrapolated_x, start_y])

    # Only prepend if extrapolated point is below p0 (i.e., y > p0[1])
    if start_y > p0[1]:
        lane_points = np.vstack([extrapolated_point, lane_points])

    # Compute arc length along the extended polyline
    deltas = np.diff(lane_points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Interpolate along arc length
    target_lengths = np.linspace(0, cumulative_lengths[-1], num_points)
    interp_x = np.interp(target_lengths, cumulative_lengths, lane_points[:, 0])
    interp_y = np.interp(target_lengths, cumulative_lengths, lane_points[:, 1])

    interpolated = np.stack((interp_x, interp_y), axis=1)
    return interpolated.astype(np.int32)
