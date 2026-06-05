// MFQ3 Gate Shader
models/mapobjects/gate/gate
{
    // Emissive cyan glow — no external light needed
    {
        map models/mapobjects/gate/gate.tga
        rgbGen identity
    }
    {
        map models/mapobjects/gate/gate.tga
        blendfunc add
        rgbGen wave sin 0.7 0.3 0 1.5
    }
}
