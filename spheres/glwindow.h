#pragma once

#define MOUSE_LEFT  0
#define MOUSE_RIGHT 1
#define NO_MOUSE    2

typedef void(*GLMouseMoveFunc) (int, int, int);

// params sets through GUI
struct GuiParams {
    int maxBounces = 10;

    float lightRadius = 500;
    float lightColor[3] = { 1, 1, 1 };
    float lightIntensity = 100;

    float skyColor[3] = { 1, 1, 1 };
    float skyIntensity = 0.2f;

    //int camera[2] = { 80, 45 }; TODO we need to expose both cam.lookFrom and cam.vUp to the UI if we want to consistently reproduce a specific view
};

void initWindow(int argc, char* argv[], int width, int height, unsigned int** _cuda_dev_render_buffer);
void registerMouseMoveFunc(GLMouseMoveFunc func);
void updateWindow(GuiParams& guiParams, bool& paramsChanged);
bool pollWindowEvents();
void destroyWindow();