#!/usr/bin/python
import sys, os, cv, Image

FACE_CASCADE = cv.Load('haarcascade_frontalface_default.xml')

def face_rects(rgba_image_f):
    """Returns a list of tuples of face rectangles, (x, y, w, h)"""
    grayscale = cv.LoadImage(rgba_image_f, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cv.EqualizeHist(grayscale, grayscale)
    faces = cv.HaarDetectObjects(grayscale, FACE_CASCADE,
                                 cv.CreateMemStorage(0), 1.2, 2,
                                 cv.CV_HAAR_DO_CANNY_PRUNING, (50,50))
    return [rect for (rect, _) in faces]


def box_in_box(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx+bw and ax+aw > bx and ay < by+bh and ay+ah > by


def faces_filter(rects):
    rects = sorted(rects, key=lambda r: r[2]*r[3])
    output = []
    for i in range(len(rects)):
        conflicted = False
        for j in range(len(output)):
            conflicted = conflicted or box_in_box(rects[i], output[j])
        if not conflicted:
            output.append(rects[i])
    return reversed(output)


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


if __name__ == "__main__":
    original = Image.open(sys.argv[1]).convert('RGBA')
    overlay = Image.open(sys.argv[2]).convert('RGBA')
    for rect in faces_filter(face_rects(sys.argv[1])):
        original = overlay_alpha_png(original, overlay, rect)
    original.save("/tmp/img.png")
