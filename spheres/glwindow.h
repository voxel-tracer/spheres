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

    bool bModelColor = false;
    float modelColor[3] = { 1, 1, 1 };
    float modelColorIntensity = 0.5f;

    bool bFixedUp = true;
};

class CudaGLContext {
public:
    const void* render_buffer; // Cuda buffer for initial render
    void* cuda_tex_resource;
    unsigned int opengl_tex_cuda;  // OpenGL Texture for cuda result

    // Texture size
    int t_width;
    int t_height;

    CudaGLContext(void* buffer, int width, int height);
};

void initWindow();
void registerMouseMoveFunc(GLMouseMoveFunc func);
void updateWindow(CudaGLContext *context, GuiParams& guiParams, bool& paramsChanged);
bool pollWindowEvents();
void destroyWindow();
