style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Japanese Anime",
        #"prompt": "anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, Linear compositions, Scalable artwork, Digital art, High Contrast Shadows",
        "prompt": "{prompt}, anime style by japanese studio, emotional, (best quality, high-res), sketch, concept art, line art, webtoon, minimal shading, high contrast, digital",

        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
      "name": "Japanese Anime1",
      "prompt": "facing the viewer, portrait shot, looking directly at camera, emotional expression, {prompt}, anime style by japanese studio, (best quality, high-res), sketch, concept art, line art, webtoon, high contrast, digital",
      "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, jpeg artifacts, signature, watermark, username, blurry"
    },
    {
        "name": "Digital/Oil Painting",
        "prompt": "{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Digital Oil Painting1",
        "prompt": "{prompt}. Extremely detailed oil painting, glow effects, god rays, hand‑drawn render, 8K Octane Cinema4D, dark atmospheric, cinematic, sharp focus, deep depth of field",
        "negative_prompt": "anime, cartoon, text, lowres, bad anatomy, extra digits, cropped, watermark, signature, artifact"
    },
    {
        "name": "Pixar Disney Character",
        "prompt": "{prompt}. Create a Disney Pixar 3D style illustration on it. The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo",
    },
    {
        "name": "Photographic",
        #"prompt": "cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus",
        "prompt": "cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, Intricate, High Detail, film photography, soft focus",
        #"negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
        "negative_prompt": "close-up, cropped face, portrait, extreme close-up, zoomed in,drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Comic book",
        "prompt": "comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed",
        "negative_prompt": "photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Line art",
        "prompt": "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Black and White Film Noir",
        "prompt": "{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Isometric Rooms",
        "prompt": "Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render",
        "negative_prompt": "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    },
    {
        "name": "Watercolor Painting",
        "prompt": "vibrant watercolor painting of {prompt}, wet-on-wet technique, soft edges, paper texture, splashes of color, trending on artstation",
        "negative_prompt": "photorealistic, 3d render, octane render, detailed, sharp focus, realism, noisy, blurry, deformed, signature, watermark"
    },
    {
        "name": "Pixel Art",
        "prompt": "8-bit pixel art of {prompt}, vibrant color palette, detailed sprites, dithering, clean outlines, trending on behance, NES style",
        "negative_prompt": "photorealistic, high resolution, smooth, gradient, blurry, 3d, photo, painting, soft lighting, detailed"
    },
    {
        "name": "Vaporwave Retrofuturism",
        "prompt": "vaporwave aesthetic of {prompt}, neon grid, palm trees, roman bust statues, retro CRT screen glow, vibrant pink and teal color palette, nostalgic, synthwave",
        "negative_prompt": "photorealistic, modern, realistic, painting, dark, monochrome, simple, ugly, deformed"
    },
    {
        "name": "Ancient Egyptian Mural",
        "prompt": "ancient egyptian mural painting of {prompt}, papyrus texture, side-profile view, hieroglyphics in the background, flat colors, gold leaf accents, fresco style",
        "negative_prompt": "3d, realistic, perspective, shading, modern, photography, portrait, blurry, deformed"
    },
    {
        "name": "Gothic Tim Burton Style",
        "prompt": "gothic illustration in the style of Tim Burton of {prompt}, dark, whimsical, spindly characters, distorted perspectives, monochrome with splashes of color, spooky, detailed ink work",
        "negative_prompt": "bright, cheerful, realistic, photorealistic, vibrant, cute, normal proportions, symmetrical, sunny"
    },
    {
        "name": "Korean Webtoon Style",
        "prompt": "korean webtoon manhwa style of {prompt}, clean line art, vibrant, cel shading, handsome characters, expressive faces, dramatic lighting, trending on naver webtoon",
        "negative_prompt": "photorealistic, painting, realistic, 3d, messy, sketch, ugly, deformed, blurry"
    },
    {
        "name": "Dieselpunk",
        "prompt": "dieselpunk artwork of {prompt}, 1940s aesthetic, heavy machinery, riveted metal, art deco design, sepia tones, gritty, industrial, analog technology",
        "negative_prompt": "futuristic, sleek, clean, modern, digital, steampunk, cyberpunk, cute, colorful, bright"
    },
    {
        "name": "Claymation",
        "prompt": "claymation style of {prompt}, stop-motion animation, plasticine texture, visible fingerprints, handcrafted, vibrant colors, Aardman Animations style, quirky",
        "negative_prompt": "photorealistic, 2d, drawing, painting, smooth, digital, clean, anime, realistic"
    },
    {
        "name": "Ukiyo-e",
        "prompt": "Ukiyo-e woodblock print style of {prompt}, in the style of Hokusai and Hiroshige, flat perspective, bold outlines, limited color palette, Japanese art, traditional",
        "negative_prompt": "photorealistic, 3d, realism, perspective, shading, modern, detailed, blurry"
    },
    {
        "name": "Board Game Art",
        "prompt": "fantasy board game box art of {prompt}, digital painting, highly rendered, saturated colors, dynamic composition, epic, detailed characters, in the style of Fantasy Flight Games",
        "negative_prompt": "photorealistic, photo, sketch, simple, minimalist, ugly, deformed, blurry, black and white"
    }
]

styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
