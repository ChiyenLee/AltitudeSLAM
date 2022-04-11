

from PIL import Image
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import exif
import utm 

def dms2decimal(dms, ref):
    """
    Convert (degrees, minutes, seconds) to decimal
    """
    decimals = dms[0] + dms[1]/60 + dms[2]/(60*60) * (-1 if ref in ["W", "S"] else 1)
    return decimals

def getGPSFromImage(filepath):
    im = exif.Image(filepath)
    long = dms2decimal(im.gps_longitude, im.gps_longitude_ref)
    lat = dms2decimal(im.gps_latitude, im.gps_latitude_ref)    
    return lat, long


if __name__ == "__main__":
    filename = "geotagged-images/IMG_0912.JPG"
    im = Image.open(filename)

    ### Extract the ground truth trajectory from KML ###
    tree = ET.parse("2013_04_14_merlischachen.kml")
    root = tree.getroot()

    # Get the Document tag and search for 
    placemark = root[0].find("{http://www.opengis.net/kml/2.2}Placemark")
    track = placemark[2]
    coord_elements = track.findall('{http://www.google.com/kml/ext/2.2}coord')
    time_elements = track.findall('{http://www.opengis.net/kml/2.2}when')

    # Data contains multipe duplicate so we need to 
    # filter them out with a hashset 
    coords = []
    time_set = set() 
    for i in range(len(coord_elements)):
        time = time_elements[i].text
        if time in time_set:
            continue
        else:
            time_set.add(time)
            coord = [float(t) for t in coord_elements[i].text.split(" ")]
            coords.append(coord)
    coords = np.array(coords)

    # ground truth from geotagged kml
    # lat long is flipped here...
    plt.plot(coords[:,1], coords[:,0], ".", label="Trajectory")

    ### Extract ground truth from GeoTag JPG ###
    counter = 0 
    latLong_list = []
    utm_list = []
    for dirpath, dnames, fnames in os.walk("geotagged-images"):
        fnames = sorted(fnames)
        for f in fnames:
            filepath = os.path.join(dirpath, f)
            im = exif.Image(filepath)
            long = dms2decimal(im.gps_longitude, im.gps_longitude_ref)
            lat = dms2decimal(im.gps_latitude, im.gps_latitude_ref)            
            
            # Convert to utm
            u = utm.from_latlon(lat, long)

            utm_list.append([u[0], u[1]])
            latLong_list.append([lat,long])
            
    latLong_list = np.array(latLong_list)
    utm_list = np.array(utm_list)
    xy_list = utm_list - utm_list[0,:]

    # plt.figure()
    # plt.plot(xy_list[:,0], xy_list[:,1])
    plt.plot(latLong_list[:,0], latLong_list[:,1], "-o", label="Keyframes")
    plt.legend()
    plt.xlabel("latitude")
    plt.ylabel("longitude")
    plt.show()