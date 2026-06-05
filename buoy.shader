// MFQ3 Waypoint Buoy Shader
models/mapobjects/waypoints/buoy
{
    // Emissive red/orange glow — no external light needed
    {
        map models/mapobjects/waypoints/buoy.tga
        rgbGen identity
    }
    {
        map models/mapobjects/waypoints/buoy.tga
        blendfunc add
        rgbGen wave sin 0.6 0.3 0 1.0
    }
}
