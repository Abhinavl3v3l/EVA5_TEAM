# JSON Description 

## First  block of json code 
``` 
{
   "001.jpg9179":{
      "filename":"001.jpg",
      "size":9179,
      "regions":[
         {
            "shape_attributes":{
               "name":"rect",
               "x":67,
               "y":36,
               "width":33,
               "height":16
            },
            "region_attributes":{
               "CLASS":"hardhat"
            }
         },
         {
            "shape_attributes":{
               "name":"rect",
               "x":60,
               "y":74,
               "width":38,
               "height":62
            },
            "region_attributes":{
               "CLASS":"vest"
            }
         },
         {
            "shape_attributes":{
               "name":"rect",
               "x":74,
               "y":204,
               "width":22,
               "height":28
            },
            "region_attributes":{
               "CLASS":"boots"
            }
         },
         {
            "shape_attributes":{
               "name":"rect",
               "x":89,
               "y":205,
               "width":34,
               "height":23
            },
            "region_attributes":{
               "CLASS":"boots"
            }
         }
      ],
      "file_attributes":{
         "caption":"",
         "public_domain":"no",
         "image_url":""
      }
   }
```

### Explanation 

* File Name is the name of the image that is annotated.
* Size refers to the memory the image takes in bytes.
* Regions contains the information about bounding boxes that were annotated on the respective image.
* shape attributes is the information about the bounding boxes we annotated on the image.
* name refers to the bounding box shape (it can be rectangle(i.e rect) or polygon or circular).
* X and Y are the bounding box starting points i.e the point where the rectangle has been started in the image making the top left corner of the image as the reference point (0,0).
* Height, width refers to the height and width of the bounding box such that it can be used to find the other opposite end of the X,Y so that it will help us to calculate the centroid of the bounding box for our clustering approach.
* Region attribute will tell us about the class of the object on which we made a bounding box.

### Approach for finding bounding boxes clusters:

<img src = "https://github.com/pruthiraj/EVA5_TEAM/blob/master/session12B/image_annotation_sample.png" alt = "img 00_1.jpg ">


* The red dots are the X,Y in the shape attributes section 
* The blue dots in the image represents the centriod of the bounding boxes we calculate using X,Y,Height,width parameters\
* These centriods are used to cluster the information and calculate the number of bounding boxes needed to cover all the images as we cannot use max number of boxes so that we are trying to reduce them by clustering the boxes making them use the nearest box.

### Cluster Graph
<img src = "https://github.com/pruthiraj/EVA5_TEAM/blob/master/session12B/cluster.png" alt = "cluster graph ">

### Cluster image
<img src = "https://github.com/pruthiraj/EVA5_TEAM/blob/master/session12B/cluster_image.png" alt = "cluster image">

