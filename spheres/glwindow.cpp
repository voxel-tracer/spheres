#ifndef GLEW_STATIC
#define GLEW_STATIC
#endif

#include "glwindow.h"

// OpenGL
#include <GL/glew.h> // Take care: GLEW should be included before GLFW
#include <GLFW/glfw3.h>
// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
//#include "cudautils.h"
#include "libs/helper_cuda.h"
// C++ libs
#include "shader_tools/GLSLProgram.h"
#include "shader_tools/GLSLShader.h"
// ImGui
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#define MATH_3D_IMPLEMENTATION
#include "math_3d.h"

// GLFW
GLFWwindow* window;
const int WIDTH = 512;
const int HEIGHT = 512;

int w_mouse_x = 0;
int w_mouse_y = 0;
bool w_mouse_left_btn = false;
bool w_mouse_right_btn = false;

GLMouseMoveFunc w_mouseMoveFunc = NULL;

// OpenGL
GLuint VBO, VAO, EBO;
GLSLShader drawtex_f; // GLSL fragment shader
GLSLShader drawtex_v; // GLSL fragment shader
GLSLProgram shdrawtex; // GLSLS program for textured draw

// Cuda <-> OpenGl interop resources

static const char* glsl_drawtex_vertshader_src =
"#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"layout (location = 2) in vec2 texCoord;\n"
"\n"
"out vec3 ourColor;\n"
"out vec2 ourTexCoord;\n"
"\n"
"void main()\n"
"{\n"
"	gl_Position = vec4(position, 1.0f);\n"
"	ourColor = color;\n"
"	ourTexCoord = texCoord;\n"
"}\n";

static const char* glsl_drawtex_fragshader_src =
"#version 330 core\n"
"uniform usampler2D tex;\n"
"in vec3 ourColor;\n"
"in vec2 ourTexCoord;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"   	vec4 c = texture(tex, ourTexCoord);\n"
"   	color = c / 255.0;\n"
"}\n";

// QUAD GEOMETRY
GLfloat vertices[] = {
    // Positions          // Colors           // Texture Coords
    1.0f, 1.0f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // Top Right
    1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // Bottom Right
    -1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // Bottom Left
    -1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f // Top Left 
};
// you can also put positions, colors and coordinates in seperate VBO's
GLuint indices[] = {  // Note that we start from 0!
    0, 1, 3,  // First Triangle
    1, 2, 3   // Second Triangle
};

void initGLBuffers()
{
    // create shader program
    drawtex_v = GLSLShader("Textured draw vertex shader", glsl_drawtex_vertshader_src, GL_VERTEX_SHADER);
    drawtex_f = GLSLShader("Textured draw fragment shader", glsl_drawtex_fragshader_src, GL_FRAGMENT_SHADER);
    shdrawtex = GLSLProgram(&drawtex_v, &drawtex_f);
    shdrawtex.compile();
}

CudaGLContext::CudaGLContext(void* buffer, int width, int height) : render_buffer(buffer), t_width(width), t_height(height) {
    // Create 2D OpenGL texture in gl_tex and bind it to CUDA in cuda_tex

    // create an OpenGL texture
    glGenTextures(1, &opengl_tex_cuda); // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, opengl_tex_cuda); // set it as current target
    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, t_width, t_height, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(
        (cudaGraphicsResource**)&cuda_tex_resource, opengl_tex_cuda, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

// Keyboard
void keyboardfunc(GLFWwindow* window, int key, int scancode, int action, int mods) {
}

void registerMouseMoveFunc(GLMouseMoveFunc func) {
    w_mouseMoveFunc = func;
}

// Mouse position
void mouseCursorPosFunc(GLFWwindow* window, double xpos, double ypos) {
    if (!w_mouseMoveFunc || ImGui::GetIO().WantCaptureMouse)
        return;
    
    int code = w_mouse_left_btn ? MOUSE_LEFT : (w_mouse_right_btn ? MOUSE_RIGHT : NO_MOUSE);
    w_mouseMoveFunc(w_mouse_x - xpos, w_mouse_y - ypos, code);

    w_mouse_x = xpos;
    w_mouse_y = ypos;
}

void mouseButtonFunc(GLFWwindow* window, int button, int action, int mods) {
    if (!w_mouseMoveFunc || ImGui::GetIO().WantCaptureMouse)
        return;

    if (action == GLFW_PRESS) {
        w_mouse_left_btn = (GLFW_MOUSE_BUTTON_LEFT == button);
        w_mouse_right_btn = (GLFW_MOUSE_BUTTON_RIGHT == button);
    }
    else if (action == GLFW_RELEASE) {
        w_mouse_left_btn = w_mouse_right_btn = false;
        w_mouseMoveFunc(0, 0, NO_MOUSE);
    }
}

bool initGL() {
    glewExperimental = GL_TRUE; // need this to enforce core profile
    GLenum err = glewInit();
    glGetError(); // parse first error
    if (err != GLEW_OK) {// Problem: glewInit failed, something is seriously wrong.
        printf("glewInit failed: %s /n", glewGetErrorString(err));
        exit(1);
    }
    glViewport(0, 0, WIDTH, HEIGHT); // viewport for x,y to normalized device coordinates transformation
    return true;
}

bool initGLFW() {
    if (!glfwInit()) exit(EXIT_FAILURE);
    // These hints switch the OpenGL profile to core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Voxel Renderer", NULL, NULL);
    if (!window) { glfwTerminate(); exit(EXIT_FAILURE); }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetKeyCallback(window, keyboardfunc);
    glfwSetCursorPosCallback(window, mouseCursorPosFunc);
    glfwSetMouseButtonCallback(window, mouseButtonFunc);
    return true;
}

void copyCUDAImageToTexture(CudaGLContext* context)
{
    // We want to copy cuda_dev_render_buffer data to the texture
    // Map buffer objects to get CUDA device pointers
    cudaArray* texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, (cudaGraphicsResource**)&context->cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, (cudaGraphicsResource*)context->cuda_tex_resource, 0, 0));

    int num_texels = context->t_width * context->t_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, context->render_buffer, size_tex_data, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, (cudaGraphicsResource**)&context->cuda_tex_resource, 0));
}

bool renderImGui(GuiParams& guiParams) {
    bool changed = false;
    ImGui::Begin("Render Options");
    if (ImGui::InputInt("max bounces", &guiParams.maxBounces))
        changed = true;
    if (ImGui::InputFloat("light radius", &guiParams.lightRadius, 1, 100))
        changed = true;
    if (ImGui::ColorEdit3("light color", guiParams.lightColor))
        changed = true;
    if (ImGui::InputFloat("light intensity", &guiParams.lightIntensity, 10, 1000))
        changed = true;

    if (ImGui::ColorEdit3("sky color", guiParams.skyColor))
        changed = true;
    if (ImGui::InputFloat("sky intensity", &guiParams.skyIntensity, 0.1f, 10))
        changed = true;
    if (ImGui::Checkbox("fixed model color", &guiParams.bModelColor))
        changed = true;
    if (ImGui::ColorEdit3("model color", guiParams.modelColor))
        changed = true;
    if (ImGui::InputFloat("model color intensity", &guiParams.modelColorIntensity, 0.1f, 10))
        changed = true;
    if (ImGui::Checkbox("fixed camera up", &guiParams.bFixedUp))
        changed = true;

    ImGui::End();

    return changed;
}

// call this after the CUDA kernel is done updating 
void updateWindow(CudaGLContext *context, GuiParams& guiParams, bool& paramsChanged) {

    copyCUDAImageToTexture(context);
    glfwPollEvents();
    // Clear the color buffer
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // feed input to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, context->opengl_tex_cuda);

    shdrawtex.use(); // we gonna use this compiled GLSL program
    glUniform1i(glGetUniformLocation(shdrawtex.program, "tex"), 0);

    glBindVertexArray(VAO); // binding VAO automatically binds EBO
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0); // unbind VAO

    // render your GUI
    paramsChanged = renderImGui(guiParams);

    // render dear imgui into the screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Swap the screen buffers
    glfwSwapBuffers(window);
}

void initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // setup platform/renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 430");
    // setup dear imgui style
    ImGui::StyleColorsDark();
}

void initWindow() {

    initGLFW();
    initGL();

    initGLBuffers();

    // Generate buffers
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // Buffer setup
    // Bind the Vertex Array Object first, then bind and set vertex buffer(s) and attribute pointer(s).
    glBindVertexArray(VAO); // all next calls wil use this VAO (descriptor for VBO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute (3 floats)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);
    // Color attribute (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    // Texture attribute (2 floats)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // Note that this is allowed, the call to glVertexAttribPointer registered VBO as the currently bound 
    // vertex buffer object so afterwards we can safely unbind
    glBindVertexArray(0);

    // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO
    // A VAO stores the glBindBuffer calls when the target is GL_ELEMENT_ARRAY_BUFFER. 
    // This also means it stores its unbind calls so make sure you don't unbind the element array buffer before unbinding your VAO, otherwise it doesn't have an EBO configured.

    initImGui();
}

// return true if user closed the window
bool pollWindowEvents() {
    glfwPollEvents();
    return glfwWindowShouldClose(window);
}

void destroyImGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void destroyWindow() {
    destroyImGui();

    glfwDestroyWindow(window);
    glfwTerminate();
}
