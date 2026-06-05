// MFQ3 Waypoint Flag Shader
models/mapobjects/waypoints/flag
{
    // Emissive cyan/orange glow — no external light needed
    {
        map models/mapobjects/waypoints/flag.tga
        rgbGen identity
    }
    {
        map models/mapobjects/waypoints/flag.tga
        blendfunc add
        rgbGen wave sin 0.6 0.3 0 1.2
    }
}
