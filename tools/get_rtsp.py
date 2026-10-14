from onvif import ONVIFCamera

ip       = "192.168.100.191"
port     = 80
user     = "admin"
password = "Pakistan2024"

try:
    cam = ONVIFCamera(ip, port, user, password)
    media = cam.create_media_service()
    profiles = media.GetProfiles()
    
    for profile in profiles:
        token = profile.token
        req   = media.create_type('GetStreamUri')
        req.ProfileToken = token
        req.StreamSetup  = {
            'Stream': 'RTP-Unicast',
            'Transport': {'Protocol': 'RTSP'}
        }
        uri = media.GetStreamUri(req)
        print(f"RTSP URL: {uri.Uri}")
        
except Exception as e:
    print(f"Error: {e}")