#!/usr/bin/python
"""
Apply transparent PNGs over faces in an image.

The overlay PNGs are square, and 3 times the size (in width and height) of
the square identified as a face, allowing for hats, beards, etc...
"""

import sys, os, cv, Image, argparse

FACE_CASCADE = cv.Load('haarcascade_frontalface_default.xml')

def box_in_box(a, b):
    """Returns True if box a and box b overlap"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx+bw and ax+aw > bx and ay < by+bh and ay+ah > by


def faces_filter(rects):
    """Filter and sort face boxes, removing overlapping boxes (keeping the smallest)"""
    rects = sorted(rects, key=lambda r: r[2] * r[3])
    output = []
    for i in range(len(rects)):
        conflicted = False
        for j in range(len(output)):
            conflicted = conflicted or box_in_box(rects[i], output[j])
        if not conflicted:
            output.append(rects[i])
    return list(reversed(output))


def face_rects(rgba_image_f):
    """Returns a list of tuples of face rectangles, (x, y, w, h)"""
    grayscale = cv.LoadImage(rgba_image_f, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cv.EqualizeHist(grayscale, grayscale)
    faces = cv.HaarDetectObjects(grayscale, FACE_CASCADE,
                                 cv.CreateMemStorage(0), 1.2, 2,
                                 cv.CV_HAAR_DO_CANNY_PRUNING, (50,50))
    return faces_filter([rect for (rect, _) in faces])


def overlay_alpha_png(rgba_image, rgba_overlay, face_rect):
    """Returns a PIL image with the alpha image on top"""
    x, y, w, h = face_rect
    c_x, c_y = x + w/2, y + h/2
    rgba_overlay = rgba_overlay.resize((w*3, h*3), Image.ANTIALIAS)
    size = rgba_overlay.size
    box = (c_x - size[0]/2,
           c_y - size[1]/2,
           c_x - size[0]/2 + size[0],
           c_y - size[1]/2 + size[1])
    rgba_image.paste(rgba_overlay, box, rgba_overlay)
    return rgba_image


def get_args():
    """Parse command line arguments"""
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument('input_image', help='Image containing face(s)')
    ap.add_argument('overlay_image', nargs='+', help='A transparent PNG to be applied to the input')
    ap.add_argument('output_image', help='The file to save the output image to')
    return ap.parse_args()


if __name__ == "__main__":
    args = get_args()
    original = Image.open(args.input_image).convert('RGBA')
    rects = face_rects(args.input_image)
    for overlay_fname in args.overlay_image:
        overlay = Image.open(overlay_fname).convert('RGBA')
        for rect in rects:
            original = overlay_alpha_png(original, overlay, rect)
    original.save(args.output_image)
