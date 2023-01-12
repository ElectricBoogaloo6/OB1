__author__ = 'Sam van Leipsig'

import pygame
import sys
import math
import Visualise_constants as vc
from reading_common import calcContrast


pygame.init()
# initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
myfont = pygame.font.Font(None, 32)
screen = pygame.display.set_mode(vc.size)
pygame.display.toggle_fullscreen()
pygame.display.set_caption("Array Backed Grid")
clock = pygame.time.Clock()

class Grid(object):
    def __init__(self,screensize,width,height,margin):
        self.screen  = screensize
        self.width = width
        self.height = height
        self.margin = margin
        self.number_rows = 10
        self.number_colums = screensize[0]/ (width+margin)
        self.grid = []
    def init_gridarray(self):
        for row in range(self.number_rows):
            self.grid.append([])
            for column in range(self.number_colums):
                self.grid[row].append(0)
    def draw(self):
        for row in range(self.number_rows):
            for column in range(self.number_colums):
                pygame.draw.rect(screen,vc.WHITE,
                             [(self.margin+self.width)*column+self.margin,
                              (self.margin+self.height)*row+self.margin,
                              self.width,self.height])


class Stimulus(object):
    def __init__(self):
        self.eyeposition = 5.
        self.stimulus = ' Beginning to read'
        self.attentional_span = 10.
        self.attentionposition = self.eyeposition
        self.fixation = 0
    def update_stimulus(self,stimulus,eyepos,att_span,attpos,fixation):
        self.stimulus = stimulus
        self.eyeposition = eyepos
        self.attentional_span = att_span
        self.attentionposition = attpos
        self.fixation = fixation+1
    def draw(self):
        if self.eyeposition != self.attentionposition:
            pygame.draw.rect(screen,vc.GREEN,
                     [(vc.margin+vc.width)*(self.attentionposition-1)+vc.margin,
                      (vc.margin+vc.height)*4+vc.margin, vc.width,vc.height])
        for pos,value in enumerate(self.stimulus):
            letter = myfont.render(value, 1, vc.BLACK, vc.WHITE)
            if self.eyeposition - pos == 0:
                pygame.draw.rect(screen,vc.YELLOW,
                         [((vc.margin+vc.width) * (pos-1)) + vc.margin,
                          ((vc.margin+vc.height) * 4) + vc.margin,
                          vc.width, vc.height])
            contrast_change = calcContrast(pos,self.eyeposition,self.attentionposition,self.attentional_span)
            letter.set_alpha(450 * contrast_change)
            letterrect = letter.get_rect()
            letterrect.centerx = (vc.margin+vc.width)* pos - (vc.width/2)
            letterrect.centery = (vc.margin+vc.height)* 5 - (vc.height/2)
            screen.blit(letter, letterrect)
    def draw_span(self):
        x = (vc.margin+vc.width)*(self.eyeposition-(self.attentional_span/2))
        y = (vc.margin+(vc.height))* 4 - (vc.height/2)
        spansize = (vc.margin+vc.width)*self.attentional_span
        pygame.draw.ellipse(screen, vc.BLACK, [x, y, spansize, 40], 1)
    def draw_fixation_number(self):
        number = myfont.render(str(self.fixation), 1, vc.BLACK, vc.WHITE)
        numberrect = number.get_rect()
        numberrect.centerx = (vc.margin+vc.width)* self.eyeposition - (vc.width/2)
        numberrect.centery = (vc.margin+vc.height)* 3 - (vc.height/2)
        screen.blit(number, numberrect)
    def draw_arrow(self):
        number = myfont.render(str("|"), 1, vc.BLACK, vc.WHITE)
        numberrect = number.get_rect()
        numberrect.centerx = (vc.margin+vc.width)* self.eyeposition - (vc.width/2)
        numberrect.centery = (vc.margin+vc.height)* 3 - (vc.height/2)
        screen.blit(number, numberrect)


## INIT
grid = Grid(vc.size,vc.width,vc.height,vc.margin)
grid.init_gridarray()
stimulus = Stimulus()

def update_stimulus(newstimulus,eyeposition,attentional_span,attentionposition,fixation):
    stimulus.update_stimulus(newstimulus,float(eyeposition),float(attentional_span),float(attentionposition),fixation)

def save_screen(fixationcounter,shift):
    pygame.image.save(screen, "Screenshots/Screen"+ str(fixationcounter) + str(shift) + ".jpg")

# -------- Main Program Loop -----------
def main():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            sys.exit()

    # Set the screen background
    screen.fill(vc.GREY)
    grid.draw()
    stimulus.draw()
    #stimulus.draw_span()
    #stimulus.draw_fixation_number()
    stimulus.draw_arrow()

    # update the screen
    clock.tick(60)
    pygame.display.update()
    #pygame.display.flip()

if __name__ == '__main__nc': main()
