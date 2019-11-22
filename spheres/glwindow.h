#pragma once

typedef void(*GLMouseMoveFunc) (int, int);

void initWindow(int argc, char* argv[], int width, int height, unsigned int** _cuda_dev_render_buffer);
void registerMouseMoveFunc(GLMouseMoveFunc func);
void updateWindow(void);
bool pollWindowEvents();
void destroyWindow();