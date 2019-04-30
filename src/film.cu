#include "film.hu"

#include "cuda_helpers.hu"

Film::Film(const FilmSettings& settings) :
    _dirty(false),
    _surface({nullptr, 0, 0}) // updateSettings frees fb and it's nop on nullptr
{
    // Generate and bind texture for blitting
    glGenFramebuffers(1 , &_fbo);
    glGenTextures(1 , &_texture);
    updateSettings(settings);
    updateTexture();
}

void Film::destroy()
{
    glDeleteFramebuffers(1 , &_fbo);
    glDeleteTextures(1 , &_texture);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(_surface.fb));
}

const Film::Surface& Film::surface() const
{
    return _surface;
}

void Film::updateSettings(const FilmSettings& settings)
{
    _surface.width = settings.width;
    _surface.height = settings.height;

    checkCudaErrors(cudaFree(_surface.fb));
    const size_t fbSize = 3 * _surface.width * _surface.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&_surface.fb), fbSize));

    // Create new texture on gpu
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _fbo);
    glBindTexture(GL_TEXTURE_2D, _texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F,_surface.width, _surface.height, 0, GL_RGB, GL_FLOAT, _surface.fb);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void Film::setDirty()
{
    _dirty = true;
}

void Film::display(uint32_t w, uint32_t h)
{
    // Update gpu texture if pixels have been updated
    if (_dirty)
        updateTexture();

    // Compensate differing aspect ratios with blit destination bounds
    // This is done to avoid stretching the render
    const float outputAspect = float(w) / h;
    const float textureAspect = float(_surface.width) / _surface.height;
    uint32_t left, top, right, bottom;
    if (outputAspect < textureAspect) {
        left = 0;
        right = w;
        const float scaledY = float(_surface.height) * w / _surface.width;
        top = (h - scaledY) / 2.f;
        bottom = top + scaledY;
    } else {
        top = 0;
        bottom = h;
        const float scaledX = float(_surface.width) * h / _surface.height;
        left = (w - scaledX) / 2.f;
        right = left + scaledX;
    }

    // Blit gpu texture to screen
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, _fbo);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, 0, _surface.width, _surface.height, left, top, right, bottom, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void Film::updateTexture()
{
    checkCudaErrors(cudaDeviceSynchronize());

    glBindTexture(GL_TEXTURE_2D, _texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _surface.width, _surface.height, GL_RGB, GL_FLOAT, _surface.fb);
    _dirty = false;
}
