import fitz
from shapely.geometry import Polygon
from pathlib import Path
from shapely.ops import cascaded_union


class PDFAnnotator():
    def __init__(self, pdf_doc, page: int):
        self.doc = pdf_doc
        self.PAGE = self.doc[page - 1]
        self.padding = 2

    def splitContent(self, content: str):
        self.content = content.split(" ")
        self.getTextRects()

    def drawShape(self, color: tuple):
        shape = self.PAGE.new_shape()
        shape.draw_polyline(self.points)
        shape.finish(color=(0, 0, 0), fill=color, stroke_opacity=0.15, fill_opacity=0.15)
        shape.commit()
        self.doc.save("[Annotated].pdf", garbage=1, deflate=True, clean=True)

    def getTextRects(self):
        rects = [self.PAGE.search_for(i) for i in self.content]  # This should produce a list of fitz.Rect
        #rects = [self.padRect(r) for r in rects]  # add padding
        polygons = self.rectToPolygon(rects[0][0])
        #polygons = [self.rectToPolygon for r in rects]  # translate fitz.Rects to shape.Polygon
        rectsMerged = cascaded_union(polygons)  # merge all polygons
        self.points = list(rectsMerged.exterior.coords)
        #self.points = polygons
        print(self.points)


    def padRect(self, rect: fitz.Rect):
        return rect + (-self.padding * 2, -self.padding, self.padding * 2, self.padding)

    def rectToPolygon(self, rect: fitz.Rect):
        upperLeft = (rect[0], rect[1])
        upperRight = (rect[2], rect[1])
        lowerRight = (rect[2], rect[3])
        lowerLeft = (rect[0], rect[3])
        return Polygon([upperLeft, upperRight, lowerRight, lowerLeft])
