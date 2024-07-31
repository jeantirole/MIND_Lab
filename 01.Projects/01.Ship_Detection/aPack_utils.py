import pandas as pd 
import datetime
import utm
import rasterio
import utm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from adjustText import adjust_text
# Visualize Shape File !!
import geopandas as gpd
from shapely.geometry import Point

import os

def filtering_datetime(df, date, time_gap_min,time_gap_sec ):
    #---- filtering datetime ---#
    
    #---- df columns 
    df.columns = ['mmsi_', 'time', 'long', 'lat', 'heading_', 'turn_', 'speed_']
        
    #---- time condition ----#
    time_gap_min 
    time_gap_sec

    earl_time = date - datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)
    post_time = date + datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)

    print("early : ",earl_time)
    print("now : ",date)
    print("late : ",post_time)

    
    earl_time = date - datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)
    post_time = date + datetime.timedelta(minutes=time_gap_min,seconds=time_gap_sec)

    print("early : ",earl_time)
    print("now : ",date)
    print("late : ",post_time)

    
     
    time_ = []
    for i,row in df.iterrows():
        #print(row["time"])
        time_.append( pd.to_datetime( row["time"] ) )
    df["datetime"] = time_

    con1 = df["datetime"] < post_time 
    con2 = df["datetime"] > earl_time

    #--- exec ---#
    time_filter_flag = True

    if time_filter_flag:
        t_df = df.loc[con1 & con2] 
        df.loc[con1 & con2]
        return df
    # else:
    #     t_df = df
    #     t_df
    
    


def coordinates_filtering(txt_path=None,xml_path=None):
    # Coordinates filtering function ! 

    if txt_path != None:
        txt_path = txt_path
        f = open(txt_path,'r')
        lines = f.readlines()
        lines

        top_left =     lines[1]
        bottom_right = lines[-1]

        min_long = float( top_left.split(":")[1].split(",")[1][0:10] )
        max_long = float( bottom_right.split(":")[1].split(",")[1][0:10] ) 

        min_lat  = float( bottom_right.split(":")[1].split(",")[0][0:10] ) 
        max_lat  = float( top_left.split(":")[1].split(",")[0][0:10] ) 

    elif xml_path != None:
        # XML parsing
        import xml.etree.ElementTree as ET

        xml_path = xml_path
        tree = ET.parse(xml_path)
        root = tree.getroot() 


        Top_left = []
        Bottom_right = []

        for child in root:
            if child.tag =="Image":
                print(child)
                for grand_child in child:
                    if grand_child.tag =="PAN":
                        for gg_child in grand_child:
                            #print(gg_child.tag)
                            if gg_child.tag == "ImagingCoordinates":
                                for ggg in gg_child:
                                    
                                    if ggg.tag =="ImageGeogTL":
                                        # print(ggg.tag)                         
                                        # print(ggg[0].text)
                                        # print(ggg[1].text)
                                        Top_left.append(float(ggg[0].text))
                                        Top_left.append(float(ggg[1].text))
                                    elif ggg.tag == "ImageGeogBR":
                                        # print(ggg.tag)                         
                                        # print(ggg[0].text)
                                        # print(ggg[1].text)
                                        Bottom_right.append(float(ggg[0].text))
                                        Bottom_right.append(float(ggg[1].text))
        
        min_long = Top_left[1]
        max_long = Bottom_right[1]
        min_lat = Bottom_right[0]
        max_lat = Top_left[0]

    # 35.032614576
    # 128.632902263
    # ImageGeogBR
    # 34.852878881
    # 128.904069499

    print(min_long,max_long,min_lat,max_lat)
    return [min_long,max_long,min_lat,max_lat]
                                    
        
def location_filtering(t_df,min_long,max_long,min_lat,max_lat ):
    con1 = t_df["long"] > min_long
    con2 = t_df["long"] < max_long

    con3 = t_df["lat"] > min_lat
    con4 = t_df["lat"] < max_lat + 0.0225 # debugged by eric 


    #------------------------------------
    location_filter_flag = True

    if location_filter_flag == True:
        #q_df = df.loc[con1 & con2 & con3 & con4]
        q_df = t_df.loc[con1 & con2 & con3 & con4]
        return q_df




def latlon_to_pixels(q_df,tf_path):
    
    #-------------------
    # utm => pixels 
    #------------------
    utm_array =[] # long lat 
    #--- csv 
    # latlong --> utm
    for lat,lon in zip(q_df['lat'].values,q_df['long'].values):
        #print(lat)
        # if lat < 80:
        res = utm.from_latlon(lat,lon,52,"N")
        utm_array.append(res)

    # 전환한 utm_array 를 데이터프레임에 병합 
    q_df['utm_array']= utm_array


    pixels_=[] # utm_array 를 pixels_ 로 변경 
    
    tf_path = tf_path
    with rasterio.open(tf_path) as map_layer:

        for inp in utm_array:
        
            coords2pixels = map_layer.index(inp[0],inp[1]) #input lon,lat # 좌표계 
            pixels_.append(coords2pixels)

    q_df['pixels'] = pixels_

    #--- init index in q_df
    q_df.drop_duplicates(inplace=True)
    q_df.reset_index(inplace=True,drop=True)

    return q_df




def match_bbox_ais(img_path,result,infer_threshold_showing,q_df,pixel_gap,img_output,dpi,save_flag):
    #------------------------------

    # Code Backup ! 

    #------------------------------
    # AIS matching with Bbox
    # Draw image 
    img_path = img_path
    img_ = Image.open(img_path)
    plt.figure(figsize=(18,18))
    #plt.imshow(img_)
    ax = plt.gca()

    # Bounding Box 
    # Thresholding filtering 
    bboxes_ = [] 
    cnt = 0
    for r in result[0]:
        if r[-1] > infer_threshold_showing:
            cnt+=1
            bboxes_.append(r)

    #----------------------------------------------------- 
    # BBox matching with AIS 

    # Matched Bbox
    idx_bboxes = []
    x_bboxes = [] 
    y_bboxes = []
    width_bboxes = []
    height_bboxes = []
    angle_bboxes = []
    prob_bboxes= []

    # Matched AIS 
    x_matched_ais = []
    y_matched_ais = []
    time_matched_ais = []
    idxss_matched_ais = []
    lat_matched_ais = []
    long_matched_ais = []

    mmsi_matched_ais = []
    heading_matched_ais = []
    turn_matched_ais = []
    speed_matched_ais = []


    # 데이터들은 q_df 써서 활용하는 것을 원칙으로 한다. 
    x_ais = [i[1] for i in q_df["pixels"].values]
    y_ais = [i[0] for i in q_df["pixels"].values]
    lat_ais  = [i for i in q_df['lat'].values]
    long_ais = [i for i in q_df['long'].values]
    mmsi_ = [i for i in q_df['mmsi_'].values]
    heading_ = [i for i in q_df['heading_'].values]
    turn_ = [i for i in q_df['turn_'].values]
    speed_ = [i for i in q_df['speed_'].values]

    idx_ais = [i for i in q_df.index.values]
    # debugged by Eric 
    time_ais = [i for i in q_df['time']]


    for bidx, box_ in enumerate( bboxes_):
        x = box_[0]
        y = box_[1]
        width =  box_[2]
        height = box_[3]
        angle_ = box_[4]
        prob_  = box_[5]
        

        # model 의 객체좌표가 center 중심인데, patches.Rectangle 은 좌하단을 기준점으로 보기 때문에. x,y 를 아래와 같이 shift 
        # roatate 할 때, 중심좌표 center 로 잡고, angle radian 변환해서 넣어주기 
        path_obj = patches.Rectangle((x-width*0.5,y-height*0.5), width, height, 
                                    linewidth=0.1, edgecolor="yellow", fill=False,
                                    rotation_point="center",
                                    angle=angle_*180/np.pi,
                                    alpha=0.35)
        
        ax.add_patch(path_obj)


        #--- find matched AIS with each box 
        cnt =0 
        for a,b,ix,t,lat_,long_,mmsi,heading,turn,speed in zip(x_ais,y_ais,idx_ais,time_ais, lat_ais,long_ais, mmsi_,heading_,turn_,speed_):
            
            # 각 bbox 범위안에 들어오는 ais 중에서 첫번쨰 ais 만 리스트에 저장 
            if cnt ==1:
                break

            # if bbox matched with ais 
            if (x - pixel_gap < a < x + pixel_gap) & (y - pixel_gap < b < y + pixel_gap) :
                
                idx_bboxes.append(bidx)
                x_bboxes.append(x)
                y_bboxes.append(y)
                width_bboxes.append(width)
                height_bboxes.append(height)
                angle_bboxes.append(angle_)
                prob_bboxes.append(prob_)

                #-- 
                x_matched_ais.append(a)
                y_matched_ais.append(b)
                time_matched_ais.append(t)
                idxss_matched_ais.append(ix)
                lat_matched_ais.append(lat_)
                long_matched_ais.append(long_)

                #-- 
                mmsi_matched_ais.append(mmsi) 
                heading_matched_ais.append(heading) 
                turn_matched_ais.append(turn) 
                speed_matched_ais.append(speed) 

                cnt+=1


    #----------------------- indent over ! 
    # 그렇다면 여기에서 통합된 bbox to ais csv 를 만들어버리자. 

    Box_Matching = True
    if Box_Matching  == True:
        T_ = pd.DataFrame({
            "box index" : idx_bboxes,
            "bbox x coord" : x_bboxes,
            "bbox y coord" : y_bboxes,
            "bbox width" : width_bboxes,
            "bbox height" : height_bboxes,
            "bbox angle" : angle_bboxes,
            "bbox prob" : prob_bboxes,
            
            #-- ais
            "mmsi" : mmsi_matched_ais,
            "heading" : heading_matched_ais,
            "turn" : turn_matched_ais,
            "speed" : speed_matched_ais,

            "ais x coord(pixel)" : x_matched_ais,
            "ais y coord(pixel)" : y_matched_ais, 
            "ais lat"   : lat_matched_ais,
            "ais long"  : long_matched_ais,
            "ais time" : time_matched_ais
        })

        T_.drop_duplicates(inplace=True)

        S_ = {
            'box index':[i for i in range(len(bboxes_)) if i not in idx_bboxes],
            'bbox x coord' : [],
            'bbox y coord' : [],
            'bbox width' : [],
            'bbox height' : [],
            'bbox angle' :[],
            'bbox prob' : [],
            "ais x coord(pixel)" : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais y coord(pixel)" : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais lat"   : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais long"  : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            "ais time" : [0 for _ in range(len([i for i in range(len(bboxes_)) if i not in idx_bboxes]))],
            }

        not_include_boxes = [i for i in range(len(bboxes_)) if i not in idx_bboxes]

        for a in not_include_boxes:
            box_ = bboxes_[a]
            
            S_['bbox x coord'].append(box_[0])
            S_['bbox y coord'].append(box_[1])
            S_['bbox width'].append(box_[2])
            S_['bbox height'].append(box_[3])
            S_['bbox angle'].append(box_[4])
            S_['bbox prob'].append(box_[5])

        S_ = pd.DataFrame(S_)

        M_ = pd.concat([T_,S_])
        #del M_["box index"]
        M_.reset_index(inplace=True,drop=True)


        #----------------
        # Draw matched AIS 
        plt.scatter(x_matched_ais,y_matched_ais, alpha=0.1,color='red',marker='o',s=8)


        #--------------------------------------------
        # among the matched AIS (q_df), label and draw them 
        plt_txts = []

            
        for i,row in M_.iterrows():
            
            a = row['ais lat']
            b = row['ais long']
            c = row['mmsi']
            d = row['heading']
            e = row['turn']
            f = row['speed']
            g = row['bbox prob']

            x = row['ais x coord(pixel)']
            y = row['ais y coord(pixel)']


            if (x ==0) & (y ==0):
                x = row['bbox x coord']
                y = row['bbox y coord']
                name = f'bbox idx : {i}, No matched AIS point'

                #-- matched 가 되지 않은 박스들에는 레이블을 붙이지 않았음 
                #plt_txts.append( plt.text(x, y, name, fontsize =2,color='yellow') )

            else:
                name = f'bbox idx : {i} mmsi : {c} heading : {d} turn : {e} speed : {f} ais coord: {a,b} bbox prob : {g}'
                plt_txts.append( plt.text(x, y, name, fontsize =2,color='red') ) 


        #----------------------------------------
        # label overlap avoid 
        adjust_text(plt_txts, 
                    force_points=3,
                    arrowprops={'arrowstyle' : '->', 
                                'color' : 'crimson',  
                                'alpha' : 0.1}
                    )
        # force_points = 3

    #---------
    # png save and draw

    if save_flag==True:
        plt.imshow(img_)
        plt.savefig(img_output,dpi = dpi )
    
        print("#--------- figure saved")
    
    return M_
        
        
def masking_bbox_land(map_path, tf_path, M_ ,csv_output_masked):
    
    # Masking을 위한 shape 불러오기 
    gdf=gpd.GeoDataFrame.from_file(map_path) #shapefile 불러오기

    # from pixel to utm
    tmp_ =[] 
    with rasterio.open(tf_path) as map_layer:
        for x,y in zip(M_['bbox x coord'].values, M_['bbox y coord'].values):
            
            x_coord,y_coord = map_layer.xy(y,x) # 여기 순서에 맞춰줘야, long,lat 좌표계가 맞음 
            tmp_.append([x_coord,y_coord])

    tmp_1= []
    ship_idxs = []
    # from utm to latlon 
    for idx,co in enumerate(tmp_):
        x,y = co[0],co[1]
        lat,lon = utm.to_latlon(x,y,52,"N")
        # print(lat,lon) # M_ 에서, 미리 설정되었던 AIS와 함께 좌표값 검증하였음

        print("progress : ", idx, " / ",len(tmp_))
        check=gdf['geometry'].contains(Point(lon,lat)) #특정 좌표 (long,lat)이 shapefile 다각형에 포함되는지 확인하기
        checkt=np.where(check == True)[0]
        
        if len(checkt) == 0:
            tmp_1.append(checkt)
            ship_idxs.append(idx)
            
    #---
    # visualize the result 
    M_masked = M_.iloc[ship_idxs]
    del M_masked['box index']
    # sav the csv 
    M_masked['box index'] = [i for i in range(len(M_masked))]
    M_masked = M_masked.reset_index(drop=True)
    #M_masked.to_csv(csv_output_masked)
    #display(M_masked)
    
    return M_masked



def visualize_mask(img_path, M_masked):
    # visualize the result with masking 
    # original image 
    
    img_ = Image.open(img_path)

    plt.figure(figsize=(18,18))
    plt.imshow(img_)
    ax = plt.gca()

    #------------------------------------
    # Bbox 
    bboxes_ = [] 
    cnt = 0
    box_cnt = 0

    # Label
    plt_txts = []

    # M_masked
    for i,row in M_masked.iterrows():
        box_idx = row['box index']
        x = row['bbox x coord']
        y = row['bbox y coord']
        width = row['bbox width']
        height = row['bbox height']
        angle_ = row['bbox angle']
        prob_ = row['bbox prob']

        mmsi_ = row['mmsi']
        heading_ = row['heading']
        turn_ = row['turn']
        speed_ = row['speed']
        lat_ = row['ais lat']
        lon_ = row['ais long']

        
        
        box_cnt+=1
        print(f"#-------------- {box_cnt / len(M_masked):.4f}")
        path_obj = patches.Rectangle((x-width*0.5,y-height*0.5), width, height, linewidth=0.1, edgecolor="red", fill=False,
                                    rotation_point="center",
                                    angle=angle_*180/np.pi)
        
        ax.add_patch(path_obj)

        # if (lat_ ==0) & (lon_ ==0):
        #     pass
        # else:
        name = f'bbox idx : {box_idx} mmsi : {mmsi_} heading : {heading_} turn : {turn_} speed : {speed_} ais coord: {lon_,lat_} bbox prob : {prob_}'
        plt_txts.append( plt.text(x, y, name, fontsize =2,color='red') ) 


    adjust_text(plt_txts, 
                force_points=3,
                arrowprops={'arrowstyle' : '->', 
                            'color' : 'crimson',  
                            'alpha' : 0.1}
                )
    #------------------------------------

    plt.imshow(img_)
    #plt.savefig(args.img_output_masked,dpi =1200 )
    plt.show()
    #------------------------------------
    
    
    
def extract_chips(M_chip, img_part,chip_folder,prefix_):
    
    #print("#--------------------",len(M_))
    for i,row in M_chip.iterrows():

        if row["ais time"] != str(0):
            print({i})
            box_ = row
            x = box_["bbox x coord"]
            y = box_["bbox y coord"]
            width =  box_["bbox width"]
            height = box_["bbox height"]
            angle_ = box_["bbox angle"]
            box_idx = box_['box index']

            pixel_slice = 512
            a_ = img_part.crop(( x-pixel_slice , y-pixel_slice, x+pixel_slice , y+pixel_slice ))
            #plt.figure(figsize=(18,18))
            
            folder_ = chip_folder
            saved_name = f"{box_idx}_{prefix_}.png"
            
            
            #plt.figure(figsize=(18,18))
            #plt.savefig(os.path.join(folder_,saved_name),dpi =600 )
            a_.save(os.path.join(folder_,saved_name))
            #plt.imshow(a_)
        


def meter_to_pixel(meter, img_size, min_long,max_long,min_lat,max_lat):
    # Multiply the degrees of separation of longitude and latitude by 111,139 to get the corresponding linear distances in meters.Nov 6, 2020

    # img size
    x_max , y_max = img_size
    # pixel => long => meter
    long_meter = 111139 * ( (max_long - min_long) / x_max )
    # pixel => lat => meter 
    lat_meter = 111139 * ( (max_lat - min_lat) / y_max ) 

    meter_per_pixel = (long_meter + lat_meter)/2

    pixel_per_meter = 1/meter_per_pixel
    
    pixel = meter * pixel_per_meter
    
    #print("meter_per_pixel : ", meter_per_pixel)
    
    return pixel