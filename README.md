# GEOM90042 Spatial Information Programming @ The University of Melbourne
## Assignment 2: Spatial Data Science: data input, manipulation analysis and presentation

COPYRIGHT 2019 The University of Melbourne.
COPYRIGHT 2019 Mustafa Neguib

## Introduction

Spatial Data Analysis is a niche in the field of data analysis in which the location aspect of a dataset such as an address of a business,
or a checkin of a user of an app is used to further enhance the analysis that has been performed. Spatial Data Analysis allows us to further 
identify and understand patterns from the data that otherwise would have gone unnoticed.

This project was an assessment for the subject Spatial Information Programming that I had studied as part of my Masters Degree.
In this project we were provided with data related to accidents in Victoria which can be found in the folder input. The description for the 
input files is given below.

## List of Files
input folder: This folder contains all of the data files that are required by this project and are as follows:
* ACCIDENT.csv: This file contains the data about accidents such as when it happened, the type of the accident where it happened, and the number of people involved.
* ACCIDENT_CHAINAGE.csv: This file contains routes in the form of chains. This file was not used in the analysis.
* ACCIDENT_EVENT.csv: This file specifies what the event was, and the part of the car where that event took place.
* ACCIDENT_LOCATION.csv: This file gives information about the road where the accident took place.
* ATMOSPHERIC_COND.csv: This file gives information about the atmospheric conditions that were at the time of the accident.
* NODE.csv: This file contains spatial information about the exact location. This file can be paired with the ACCIDENT_LOCATION.csv
* NODE_ID_COMPLEX_INT_ID.csv
* PERSON.csv: This file gives demographics information about the people who were involved in the accident.
* ROAD_SURFACE_COND.csv: This file gives information about the condition of the road at the time of the accident.
* SUBDCA.csv
* VEHICLE.csv: This file gives information about the vehicles which were involved in the accident.
* Statistic Checks.csv
* SA2_2016_AUST shape files: These files are readable in GIS software and geopandas and have been used for spatial analysis which include relational geometry as well.



http://data.vicroads.vic.gov.au/metadata/crashstats_user_guide_and_appendices.pdf
http://www.abs.gov.au/ausstats/abs@.nsf/Latestproducts/88F6A0EDEB8879C0CA257801000C64D9

output folder: This folder contains all the files and data that this project generates as a result.

The following are the notebook and the module Python file that are executable.
assign2.ipynb
assign2_modules.py


The HTML webpages of the report that were generated have been converted into a pdf format and the name of the file is 
Spatial Information Programming Final Report.pdf


## Work Done So Far
This project was analyzed in Python 3.6 Jupyter Notebook and used the related spatial analysis libraries.
The following analysis have already been performed on the dataset:
* Preprocessing & transformation of the data: Preprocessing tasks and transformation on the data were performed in order to analyse the dataset. This is required so that the data is cleaned up and any errors are removed which may negatively affect the final analysis.
* Time of day and day Analysis: This allowed us to analyse and understand the time at which the accidents took place. This is important in order to know at what time of the day are accidents most likely to happen.
* Spatial Temporal Visual Analysis 
* Demographic Analysis: In order to analyse why accidents happen we must understand who are actually involved in these events so that we can identify at risk people.
* Clustering Analysis: This allows us to see how distributed the accidents are, whether they are concentrated in certain locations or are they far apart. This analysis helps us identify those locations which have a high density of accidents.
* Visualization analysis: Visualization of the number of accidents by suburbs helps us identify those suburbs where the number of accidents are the highest.
* Spatial Autocorrelation: This analysis allows us to study whether the number of accidents in the suburbs are clustered or are they randomly distributed. If there is a certain clustering then we can further identify at risk locations.
* Identification of the road/street where the most number of accidents took place: This is an effort to granularize and find the most at risk roads where the most number of accidents have taken place.
* Identification of falling trend in the number of accidents over the time period: This was an effect of the study performed and was quite surprising to see that in fact there was a decline in the number of accidents over the years.
