"""
This project has been developed by Mustafa Neguib who is a student of
Masters of Information Technology at The University of Melbourne and has
student id 922939.
You can contact the developer at mneguib@student.unimelb.edu.au
"""
import os
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from dateutil.parser import parse
from shapely import geometry
from rtree import index
import libpysal as lps
import sklearn.preprocessing
from sklearn.cluster import DBSCAN
from sklearn import metrics
import mapclassify as mc
import esda
import seaborn as sbn
import random
import folium


def read_csv_file(csv_file_name, required_columns):
    """
    This function reads in the csv file and then checks if the file has the correct
    number of columns and also if the column names are correct.
    :param csv_file_name: A string containing the name of the csv file to read.
    :param required_columns: A list containing the column names that are required by this project for the file.
    :return: data_frame is a pandas data frame which contains the data read from the csv file

    """

    execution_halted_str = 'Execution halted in the function read_csv_file!!!'
    path_of_file = os.path.join(os.getcwd(), csv_file_name)
    if os.path.exists(path_of_file):
        data_frame = pd.read_csv(path_of_file)

        if data_frame.columns.size == len(required_columns):

            # This code is checking if the column exists, no matter in what order it comes in the
            # csv file. This means that as long as the column exists in the csv file we do not care,
            # whether it is the first column, or the last column.

            found_invalid_columns = False
            invalid_columns = []
            for column in required_columns:
                try:
                    # If the column is not in the data frame, then a ValueError will be thrown, which
                    # will tell me that there was a problem. If no such error is thrown then the data frame
                    # has all of the required columns.
                    list(data_frame.columns).index(column)
                except ValueError as value_error:
                    found_invalid_columns = True
                    invalid_columns.append(column)

            if found_invalid_columns:
                raise Exception(
                    '{} The csv file {} provided does not contain the column(s) {}.'.format(execution_halted_str,
                                                                                            csv_file_name,
                                                                                            ', '.join(invalid_columns)))
            return data_frame
        else:
            raise Exception(
                "{} The csv file {} does not contain all of the required columns. Please check the file and ensure that the columns and their names are correct.".format(
                    execution_halted_str, csv_file_name))
    else:
        raise FileNotFoundError(
            "{} The csv file {} was not found in the current directory.".format(execution_halted_str, csv_file_name))


def read_shape_file(shape_file_name, required_columns):
    """
    This function reads in the provided shape file from the disk
    :param shape_file_name:  A string containing the name of the shape file to read.
    :param required_columns: A list containing the column names that are required by this project for the file.
    :return: geo_data_frame is a geopandas data frame which contains the data read from the shape file
    """

    execution_halted_str = 'Execution halted in the function read_shape_file!!!'
    path_of_file = os.path.join(os.getcwd(), shape_file_name)
    if os.path.exists(path_of_file):

        # This code is checking if the column exists, no matter in what order it comes in the
        # csv file. This means that as long as the column exists in the csv file we do not care,
        # whether it is the first column, or the last column.

        geo_data_frame = geopandas.read_file(shape_file_name)
        if geo_data_frame.columns.size == len(required_columns):

            # This code is checking if the column exists, no matter in what order it comes in the
            # csv file. This means that as long as the column exists in the csv file we do not care,
            # whether it is the first column, or the last column.

            found_invalid_columns = False
            invalid_columns = []
            for column in required_columns:
                try:
                    # If the column is not in the data frame, then a ValueError will be thrown, which
                    # will tell me that there was a problem. If no such error is thrown then the data frame
                    # has all of the required columns.
                    list(geo_data_frame.columns).index(column)
                except ValueError as value_error:
                    found_invalid_columns = True
                    invalid_columns.append(column)

            if found_invalid_columns:
                raise Exception(
                    '{} The shape file {} provided does not contain the column(s) {}.'.format(execution_halted_str,
                                                                                              shape_file_name,
                                                                                              ', '.join(
                                                                                                  invalid_columns)))
            return geo_data_frame
        else:
            raise Exception(
                "{} The shape file {} does not contain all of the required columns. Please check the file and ensure that the columns and their names are correct.".format(
                    execution_halted_str, shape_file_name))

    else:
        raise FileNotFoundError(
            "{} The shape file {} was not found in the current directory.".format(execution_halted_str,
                                                                                  shape_file_name))


# The following functions are for driving the questions in Task 1

def calculate_accident_statistics(accidents_data_frame):
    """
    This function calculates the statistics about accidents by vehicle and years and then
    returns an html string containing the formatted text with the appropriate values,
    and also returns a pandas data frame which contains data of accidents by vehicles
    which is then used by other functions.

    :param accidents_data_frame: data frame containing data related to accidents
    :param vehicles_data_frame: data frame containing data related to vehicles that were in the accident
    :return: html_string, accidents_data_frame_few_columns
    """
    accidents_data_frame_few_columns = get_columns_from_data(accidents_data_frame,
                                                             ['ACCIDENT_NO', 'ACCIDENTDATE', 'Accident Type Desc'])

    # Get the all of the rows from the data frame accidents_with_vehicles which are not from 2017
    # The False flag tells the function to ignore the rows that are from 2017.

    accidents_data_frame_few_columns = get_data_filtered_accident_date_for_year(accidents_data_frame_few_columns,
                                                                                '2017', False)

    # I am extracting the years and saving them in a set, so that i have the exact years when the
    # accidents took place.

    # If the date is of the wrong format, do not select the date.
    accidents_data_frame_few_columns['YEAR'] = accidents_data_frame_few_columns.ACCIDENTDATE.apply(
        lambda x: str(parse(x).year))
    grouped_by_year_accidents_data_frames = group_by_data(accidents_data_frame_few_columns, ['YEAR'])
    average_num_accidents_per_year = grouped_by_year_accidents_data_frames['count'].values.mean()

    print("Average number of accidents per year: {:.2f}".format(average_num_accidents_per_year))
    html_content = build_html_component(
        "Average number of accidents per year: {:.2f}".format(average_num_accidents_per_year),
        "Question 1 -- Calculating the average number of accidents per year")

    num_accidents_by_type = accidents_data_frame_few_columns["Accident Type Desc"].value_counts().sort_values(
        ascending=False)

    # I want to get the second most common type of accident so that is why i want to get the element at index 1,
    # as the element at index 0 is the most common type of accident. The titles for the accident type are being used
    # as the index, but i am accessing the series by using integer indexing.
    second_most_common_type = ''
    if num_accidents_by_type.shape[0] >= 2 and num_accidents_by_type.sum() > 0:
        percent_accidents = (num_accidents_by_type[1] / num_accidents_by_type.sum()) * 100
        second_most_common_type = num_accidents_by_type.index[1]
    else:
        percent_accidents = 0

    print(
        "The second most common type of accident in all the recorded years is '{}', and the percentage of the accidents that belong to this type is {:.2f}%".format(
            second_most_common_type, percent_accidents))

    return html_content + build_html_component(
        "The second most common type of accident in all the recorded years is '{}', and the percentage of the accidents that belong to this type is {:.2f}%".format(
            second_most_common_type, percent_accidents),
        "Question 2 -- Number and proportion of the second most common type of accidents"), accidents_data_frame_few_columns


def generate_accidents_by_type_and_year(accidents_data_frame_few_columns, vehicles_data_frame):
    """
    This function generates a table for accident data which is listed by
    teh vehicle type and year. The return value is a HTML string which is
    the table formatted as a HTML table.
    :param accidents_with_vehicles:
    :return: html_string, accidents_with_vehicles
    """
    vehicles_data_frame_few_columns = get_columns_from_data(vehicles_data_frame, ['ACCIDENT_NO', 'Vehicle Type Desc'])
    accidents_with_vehicles = pd.merge(accidents_data_frame_few_columns, vehicles_data_frame_few_columns,
                                       on="ACCIDENT_NO",
                                       how="inner")
    accidents_with_vehicles = get_data_filtered_accident_date_for_year(accidents_with_vehicles, '2017', False)

    accidents_with_vehicles['YEAR'] = accidents_with_vehicles.ACCIDENTDATE.apply(lambda x: str(parse(x).year))
    grouped = group_by_data(accidents_with_vehicles, ['Vehicle Type Desc', 'YEAR'])
    pivoted_table = grouped.pivot(index='Vehicle Type Desc', columns='YEAR', values='count')
    pivoted_table.fillna(0, inplace=True)
    pivoted_table.to_csv('output/AccidentByYear.csv')

    return build_html_component(pivoted_table.to_html(),
                                "Question 3 -- Number of accidents by vehicle type by year"), accidents_with_vehicles


def compute_top10_lga_accidents(accidents_data_frame, node_data_frame):
    """
    This function is computing the top 10 LGA areas where the most accidents have occurred.

    :param accidents_data_frame:
    :param node_data_frame:
    :return: html_string containing the html table
    """

    # I am exctracting accidents_data_frame_few_columns contains only those columns that have been specified. This
    # allows me to have data frames of smaller widths which only have the information that I need.
    accidents_data_frame_few_columns = get_columns_from_data(accidents_data_frame,
                                                             ['ACCIDENT_NO', 'ACCIDENTDATE', 'NODE_ID'])
    node_data_frame_few_columns = get_columns_from_data(node_data_frame, ['ACCIDENT_NO', 'NODE_ID', 'LGA_NAME'])
    # I am doing an inner join on the two data frames. This is basically analogous to sql inner join.
    accidents_with_lgas = pd.merge(accidents_data_frame_few_columns, node_data_frame_few_columns,
                                   on=["ACCIDENT_NO", "NODE_ID"], how="inner")
    # Give me those rows that are of 2006 only
    accidents_with_lgas_2006 = get_data_filtered_accident_date_for_year(accidents_with_lgas, '2006', True)

    # Give me those rows that are of 2016 only
    accidents_with_lgas_2016 = get_data_filtered_accident_date_for_year(accidents_with_lgas, '2016', True)
    grouped_lgas_2006 = group_by_data(accidents_with_lgas_2006, ['LGA_NAME'])
    grouped_lgas_2006.sort_values(by='count', inplace=True, ascending=False)
    grouped_lgas_2006 = grouped_lgas_2006.reset_index(drop=True)

    # This is returning the top 10 LGAs for 2006 and have been sorted according to the most number of accidents in 2006
    grouped_lgas_2006 = grouped_lgas_2006[0:10]
    grouped_lgas_2016 = group_by_data(accidents_with_lgas_2016, ['LGA_NAME'])

    # merged_2006_2016_lgas contains data for the LGAs for 2006 and also 2016
    merged_2006_2016_lgas = pd.merge(grouped_lgas_2006, grouped_lgas_2016, on=["LGA_NAME"], how="inner")
    merged_2006_2016_lgas['NUM_2006'] = merged_2006_2016_lgas['count_x']
    merged_2006_2016_lgas['NUM_2016'] = merged_2006_2016_lgas['count_y']
    merged_2006_2016_lgas = get_columns_from_data(merged_2006_2016_lgas, ['LGA_NAME', 'NUM_2006', 'NUM_2016'])
    merged_2006_2016_lgas['DIFFERENCE'] = merged_2006_2016_lgas['NUM_2016'] - merged_2006_2016_lgas[
        'NUM_2006']
    merged_2006_2016_lgas['CHANGE'] = ((merged_2006_2016_lgas['NUM_2016'] - merged_2006_2016_lgas[
        'NUM_2006']) / merged_2006_2016_lgas['NUM_2006']) * 100
    merged_2006_2016_lgas.to_csv('output/AccidentByLGA.csv')

    return build_html_component(merged_2006_2016_lgas.to_html(),
                                "Question 4 -- Top 10 local government areas")


def plot_accidents_by_day_week(accidents_data_frame):
    """
    This function prepares the data of accidents by day week so that it can be plotted.
    :param accidents_data_frame: data frame containing information about accidents
    :return: html_string
    """
    accidents_data_frame_few_columns = get_columns_from_data(accidents_data_frame,
                                                             ['ACCIDENT_NO', 'ACCIDENTDATE', 'Day Week Description'])

    accidents_day_week_2006 = get_data_filtered_accident_date_for_year(accidents_data_frame_few_columns, '2006', True)
    accidents_day_week_2016 = get_data_filtered_accident_date_for_year(accidents_data_frame_few_columns, '2016', True)
    grouped_day_week_2006 = group_by_data(accidents_day_week_2006, ['Day Week Description'])
    grouped_day_week_2016 = group_by_data(accidents_day_week_2016, ['Day Week Description'])

    # The variables days_2006, num_days_2006, days_2016, and num_days_2016 are all lists containing the number
    # of accidents and the labels of the days of the week when those accidents took place
    days_2006_labels = grouped_day_week_2006['Day Week Description'].values
    num_days_2006 = grouped_day_week_2006['count'].values
    days_2016_labels = grouped_day_week_2016['Day Week Description'].values
    num_days_2016 = grouped_day_week_2016['count'].values

    image_link = build_bar_chart_with_two_bars_per_label(num_days_2006, num_days_2016, '2006', '2016', days_2006_labels,
                                                         days_2016_labels,
                                                         'Accident numbers by days of the week, in 2006 and 2016',
                                                         'Years', 'No of Accidents', 'accidents_by_day_week.jpg')

    image_link = """
            <img src='{0}' style='width:50%;' />
    """.format(image_link)
    return build_html_component(image_link, "Question 5 -- Accident numbers by days of the week in 2006 and 2016")


def assign_severity_description(x):
    """
    This function returns aa textual representation of the severity as a string.
    :param x: integer value being one of the values for severity
    :return: string representation of the integer value
    """

    if x['SEVERITY'] == 1:
        return "Fatal accident"
    elif x['SEVERITY'] == 2:
        return "Serious injury accident"
    elif x['SEVERITY'] == 3:
        return "Other injury accident"
    elif x['SEVERITY'] == 4:
        return "Non injury accident"
    else:
        return "Non injury accident"


def plot_accidents_by_severity_year(accidents_data_frame):
    """
    This function prepares the data of accidents by severity and year so that it can be plotted.
    :param accidents_data_frame:
    :return: html_string
    """
    accidents_data_frame_few_columns = get_columns_from_data(accidents_data_frame,
                                                             ['ACCIDENT_NO', 'ACCIDENTDATE', 'SEVERITY'])
    accidents_data_frame_few_columns['Severity Description'] = accidents_data_frame_few_columns.apply(
        assign_severity_description, axis=1)
    data_list_by_years = []
    for i in range(2006, 2017):
        data_list_by_years.append(
            get_data_filtered_accident_date_for_year(accidents_data_frame_few_columns, str(i), True))

    data_list_by_years_grouped = []
    for i in data_list_by_years:
        data_list_by_years_grouped.append(group_by_data(i, ['SEVERITY', 'Severity Description']))

    # These definitions have been given in the crash stats document
    severity_lists_descriptions = ['Fatal accident', 'Serious injury accident', 'Other injury accident',
                                   'Non injury accident']
    data_by_years = []
    for data in data_list_by_years_grouped:
        d = []
        if data.shape[0] == len(severity_lists_descriptions):
            d = data.values
        else:
            diff_length = len(severity_lists_descriptions) - data.shape[0]
            d = data.values
            for j in range(diff_length):
                index = len(severity_lists_descriptions) - j
                severity = severity_lists_descriptions[index - 1]
                d = np.append(d, [[int(data.shape[0] + (j + 1)), severity, int(0)]], 0)

        data_by_years.append(d)

    change = []
    for j in range(1, len(data_by_years)):
        change_in_severity = []
        if int(data_by_years[j - 1][0][2]) == 0:
            change_in_severity.append(0)
        else:
            change_in_severity.append(((int(data_by_years[j - 1][0][2]) - int(data_by_years[j][0][2])) / int(
                data_by_years[j - 1][0][2])) * 100)

        if int(data_by_years[j - 1][1][2]) == 0:
            change_in_severity.append(0)
        else:
            change_in_severity.append(((int(data_by_years[j - 1][1][2]) - int(data_by_years[j][1][2])) / int(
                data_by_years[j - 1][1][2])) * 100)

        if int(data_by_years[j - 1][2][2]) == 0:
            change_in_severity.append(0)
        else:
            change_in_severity.append(((int(data_by_years[j - 1][2][2]) - int(data_by_years[j][2][2])) / int(
                data_by_years[j - 1][2][2])) * 100)

        if int(data_by_years[j - 1][3][2]) == 0:
            change_in_severity.append(0)
        else:
            change_in_severity.append(((int(data_by_years[j - 1][3][2]) - int(data_by_years[j][3][2])) / int(
                data_by_years[j - 1][3][2])) * 100)
        change.append(change_in_severity)

    image_link = build_line_chart_with_multiple_categories(change, severity_lists_descriptions,
                                                           2006,
                                                           2016,
                                                           'Yearly change of the total number of accidents from 2006 to 2016',
                                                           'Years', 'Percentage Change in Number of Accidents',
                                                           'yearly_change_tot_num_2006_to_2016.jpg')

    image_link = """
            <img src='{0}' style='width:50%;' />
    """.format(image_link)
    return build_html_component(image_link,
                                "Question 6 -- Yearly change of the number of accidents from 2006 to 2016 for each severity category")


def get_data_filtered_accident_date_for_year(data, year, condition_boolean):
    """
    This function filters out the data against the year that has been provided, based on the boolean value that is
    given.
    :param data: pandas data frame which has the column for the date
    :param year: a string value for the year
    :param condition_boolean: either True or False
    :return: a filtered pandas data frame either containing the rows for the year provided or excluding the years
    depending on the value of the condition_boolean parameter
    """

    return data[data.ACCIDENTDATE.str.contains(year) == condition_boolean].reset_index(drop=True)


def group_by_data(data, columns):
    """
    This is a wrapper function which wraps around the pandas group by function.
    :param data: Pandas data frame
    :param columns: The columns which are to be used to be group the data frame on
    :return: Pandas data frame
    """
    return data.groupby(columns).size().to_frame('count').reset_index()


def get_columns_from_data(data, columns):
    """
    This function is a wrapper around the indexing of pandas data frames. This function gets all of the rows
    from the data frame.
    :param data: Pandas data frame
    :param columns: The columns which are to be selected.
    :return: Pandas data frame
    """
    return data.loc[:, columns]


def get_columns_from_data_range_rows(data, start_row, end_row, columns):
    """
    This function is a wrapper around the indexing of pandas data frames. This function gets some of the rows
    from the data frame specified by the start_row and end_row.
    :param data: Pandas data frame
    :param start_row: int value specifying the row to start from
    :param end_row: int value specifying the row to end at
    :param columns: The columns which are to be selected.
    :return: Pandas data frame
    """
    return data.loc[start_row:end_row, columns]


def build_bar_chart_with_two_bars_per_label(series1, series2, series1_label, series2_label, series1_labels,
                                            series2_labels,
                                            title, x_axis_label, y_axis_label, output_file_name):
    """
    This function builds a bar chart that has two bars per label.
    :param series1: a list of values containing the data for the first series
    :param series2: a list of values containing the data for the second series
    :param series1_label: a label to be shown in the legend for the first series
    :param series2_label: a label to be shown in the legend for the second series
    :param series1_labels: a list of labels for the first series
    :param series2_labels: a list of labels for the second series
    :param title: string value of the title of the bar chart
    :param x_axis_label: the label to show on the x axis
    :param y_axis_label: the label to show on the y axis
    :param output_file_name: the name and path of the file where the figure is to be exported to
    :return: string path of the image that has been saved of the figure
    """
    index_series1 = np.arange(len(series1_labels))
    index_series2 = np.arange(len(series2_labels))
    fig, ax = plt.subplots()
    ax.bar(x=index_series1 - 0.4, height=series1, width=0.4, bottom=0, align='center', label=series1_label)
    ax.bar(x=index_series2, height=series2, width=0.4, bottom=0, align='center', label=series2_label)
    ax.set_xlabel(x_axis_label, fontsize=10)
    ax.set_ylabel(y_axis_label, fontsize=10)
    ax.set_xticks(index_series1)
    ax.set_xticklabels(series1_labels, fontsize=10, rotation=30)
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True)
    plt.show()
    output_file_name = "output/" + output_file_name
    fig.savefig(output_file_name, dpi=300, bbox_inches='tight')
    return '../{}'.format(output_file_name)


def build_line_chart_with_multiple_categories(data, categories, start_index,
                                              end_index, title, x_axis_label,
                                              y_axis_label, output_file_name):
    # This function builds a line chart.
    """

    :param data: a list containing the data that is to be plotted
    :param categories: list string values containing the categories
    :param start_index: int value starting index of the x-axis
    :param end_index: int value ending index of the x-axis
    :param title: string value of the title of the chart
    :param x_axis_label: string value of the label of x-axis
    :param y_axis_label: string value of the label of the y-axis
    :param output_file_name: string value of the file name where the resulting figure is to be saved
    :return: returns the string value of the path where the resulting figure has been saved
    """
    fig, ax = plt.subplots()
    index = np.arange(end_index - start_index)

    x_labels = []
    for i in range(start_index, end_index):
        x_labels.append(str(i))
    # This list will be a 2D list. The first dimension will contain a list of severities (4 in our case) and then
    # each element of that will contain 11 elements where each element is a year from 2006 to 2016.
    # These elements will contain the number of accidents that took place in that year for that severity

    lines = []

    for i in range(0, len(categories)):
        lines.append([])

    for i in range(0, len(categories)):
        for j in range(0, end_index - start_index):
            lines[i].append(0)

    for i in range(0, len(categories)):
        for j in range(0, end_index - start_index):
            lines[i][j] = data[j][i]

    for i, data in enumerate(lines):
        ax.plot(index, data, label=categories[i])

    ax.set_xlabel(x_axis_label, fontsize=10)
    ax.set_ylabel(y_axis_label, fontsize=10)
    ax.set_xticks(index)
    ax.set_xticklabels(x_labels, fontsize=10, rotation=30)
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True)
    plt.show()
    output_file_name = "output/" + output_file_name
    fig.savefig(output_file_name, dpi=300, bbox_inches='tight')
    return '../{}'.format(output_file_name)


def build_html_string(html_string, page_title):
    """
    This function builds the HTML document inside of it is placed the HTML components that have been built by other
    functions.
    :param html_string:
    :return:
    """
    html = """
    <html>
     <head>
            <title>{0}</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">            
            <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
            <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
     </head>
       <body>
         <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
          <a class="navbar-brand" href="#">Spatial Information Programming Assignment 2 {0}</a>
          <div class="collapse navbar-collapse" id="navbarsExampleDefault">
            <ul class="navbar-nav mr-auto">
              <li class="nav-item active">
                <a class="nav-link" href="#">Home <span class="sr-only">(current)</span></a>
              </li>
            </ul>
          </div>
        </nav>
        <main role="main" class="container" style="margin-top: 80px;">
         <div class="row">
            <div class="col-md-12">
                <h5>
                    Project and report developed by Mustafa Neguib with student number 922939
                </h5>
            </div>
          </div>
          {1}
          </main>  
       </body>    
    </html>
    """.format(page_title, html_string)
    return html


def build_html_component_with_html(html_string):
    """
    This function builds the html string for a component.
    :param html_string: html_string of the component
    :param title: Title of the html component
    :return: html_string
    """
    html = """
    
    <div class="row">
        <div class="col-md-12">
            {0}
        </div>
    </div>
    
    """.format(html_string)

    return html


def build_html_component_without_title(html_string):
    """
    This function builds the html string for a component.
    :param html_string: html_string of the component
    :param title: Title of the html component
    :return: html_string
    """
    html = """
    
    <div class="row">
        <div class="col-md-12">
            <p>
            {0}
            </p>
        </div>
    </div>
    
    """.format(html_string)

    return html


def build_html_component(html_string, title):
    """
    This function builds the html string for a component.
    :param html_string: html_string of the component
    :param title: Title of the html component
    :return: html_string
    """
    html = """
    
    <div class="row">
        <div class="col-md-12">
            <h5>
            {1}
            </h5>
            <p>
            {0}
            </p>
        </div>
    </div>
    
    """.format(html_string, title)

    return html


def write_to_file(html_content, file_name, page_title):
    file_name = "output/" + file_name
    path = os.path.join(os.getcwd(), file_name)
    with open(path, 'w') as writeFile:
        writeFile.write(build_html_string(html_content, page_title))


def build_geometry(data):
    """
    This function builds a Shapely Point geometry based on the longitude and latitude values.
    :param data: Pandas data frame
    :return: Shapely Point geometry
    """
    return geometry.Point(data['Long'], data['Lat'])


# The following functions are for driving the questions in Task 2

def build_accident_locations_shape_file(accidents_data_frame, vehicles_data_frame, node_data_frame):
    """
    This function builds the shape file which contains the locations of accidents and the related data.
    :param accidents_data_frame: Pandas data frame containing information about accidents
    :param vehicles_data_frame: Pandas data frame containing information about vehicles
    :param node_data_frame: Pandas data frame containing information about nodes
    """
    accidents_data = get_columns_from_data(accidents_data_frame, ['ACCIDENT_NO', 'Day Week Description', 'NODE_ID'])
    vehicles_data = get_columns_from_data(vehicles_data_frame, ['ACCIDENT_NO', 'Vehicle Type Desc'])
    node_data = get_columns_from_data(node_data_frame, ['ACCIDENT_NO', 'NODE_ID', 'Lat', 'Long'])
    accidents_data = pd.merge(accidents_data, vehicles_data,
                              on="ACCIDENT_NO",
                              how="inner")
    accidents_data = pd.merge(accidents_data, node_data,
                              on=["ACCIDENT_NO", "NODE_ID"], how="inner")
    grouped = accidents_data.groupby(['ACCIDENT_NO', 'Day Week Description', 'NODE_ID', 'Lat', 'Long']).size().to_frame(
        'count').reset_index()
    grouped_listed_veh_type = accidents_data.groupby('ACCIDENT_NO')['Vehicle Type Desc'].agg(list)
    accidents_locations = pd.merge(grouped, grouped_listed_veh_type,
                                   on="ACCIDENT_NO",
                                   how="inner")
    accidents_locations['Vehicle Type Desc'] = accidents_locations['Vehicle Type Desc'].apply(lambda x: ', '.join(x))
    accidents_locations['SevereAccident'] = accidents_locations['count'].apply(lambda x: 1 if x >= 3 else 0)
    accidents_locations = accidents_locations.rename(index=str, columns={"ACCIDENT_NO": "AccidentNumber",
                                                                         "Day Week Description": "DayOfWeek",
                                                                         "Vehicle Type Desc": "VehicleType"})

    accidents_locations = accidents_locations.drop(['NODE_ID', 'count'], axis=1)
    accidents_locations['geometry'] = accidents_locations.apply(build_geometry, axis=1)
    accidents_locations = accidents_locations.drop(['Lat', 'Long'], axis=1)
    accidents_locations = geopandas.GeoDataFrame(accidents_locations, geometry=accidents_locations.geometry)
    accidents_locations.crs = {'init': 'epsg:4326', 'no_defs': True}
    accidents_locations.to_file('output/AccidentsLocation.shp')


def build_weekday_weekend_shape_files(accidents_locations):
    severe_accident_weekday = accidents_locations[
        (accidents_locations.DayOfWeek.isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])) & (
                accidents_locations.SevereAcci == 1)]
    severe_accident_weekend = accidents_locations[
        (accidents_locations.DayOfWeek.isin(['Saturday', 'Sunday'])) & (accidents_locations.SevereAcci == 1)]
    severe_accident_weekday.reset_index(inplace=True)
    severe_accident_weekday = severe_accident_weekday.drop('index', axis=1)
    severe_accident_weekend.reset_index(inplace=True)
    severe_accident_weekend = severe_accident_weekend.drop('index', axis=1)
    severe_accident_weekday.to_file('output/SevereAccidentWeekday.shp')
    severe_accident_weekend.to_file('output/SevereAccidentWeekend.shp')


def add_sa2_names_to_accidents_locations_shape_file_sjoin(sa2):
    """
    This function does quite a number of things from transforming the SA2 geometries to 4326 crs from 4283 crs, then
    extracts the geometries based on the bounding box of the accidents_locations geometries. This function and
    the next function add_sa2_names_to_accidents_locations_shape_file are returning the same results but through
    different ways. This function is extremely fast because the sjoin function has been used which joins the two
    tables like two tables are joined in relational database. This function takes about 10 seconds to run, so for
    practical reasons this function is being called and not the other one in the notebook file.
    :param sa2: geopandas geodataframe file containing the data of SA2 areas
    :return:  accidents_locations, precise_matches accidents_locations and sa2_areas_of_interest are geopandas
    geodataframes. accidents_locations contains data related to the accident along with the name of the SA2 where the
    accident took place. sa2_areas_of_interest are those SA2 where the accidents may have taken place.
    """
    required_columns = ['AccidentNu', 'DayOfWeek', 'VehicleTyp', 'SevereAcci', 'geometry']
    accidents_locations = read_shape_file('output/AccidentsLocation.shp', required_columns)
    # The SA2 geodataframe has geometries which had null as values, and due to those values we were not able to
    # transform the crs of the geometries, so we are removing the rows with null geometries.
    sa2 = sa2.loc[sa2['geometry'].notnull()]
    sa2 = sa2.to_crs(epsg=4326)

    bounds = accidents_locations.geometry.total_bounds
    min_x = bounds[0] - 1
    min_y = bounds[1] - 1
    max_x = bounds[2] + 1
    max_y = bounds[3] + 1

    sa2_areas_of_interest = sa2[(sa2.geometry.bounds['minx'] >= min_x) & (sa2.geometry.bounds['maxx'] <= max_x) & (
            sa2.geometry.bounds['miny'] >= min_y) & (sa2.geometry.bounds['maxy'] <= max_y)]

    accidents_locations = geopandas.sjoin(accidents_locations, sa2, how='left', op='within')
    accidents_locations = accidents_locations.rename(index=str, columns={'SA2_NAME16': 'SA2'})
    return get_columns_from_data(accidents_locations,
                                 ['AccidentNu', 'DayOfWeek', 'VehicleTyp', 'SevereAcci', 'geometry',
                                  'SA2']), sa2_areas_of_interest


def add_sa2_names_to_accidents_locations_shape_file(sa2):
    """
    This function does quite a number of things from transforming the SA2 geometries to 4326 crs from 4283 crs, then
    extracts the geometries based on the bounding box of the accidents_locations geometries. Then rtree index is used
    to further filter and refine the search, and then saving the name of the SA2 are into a new column in the
    accidents_locations.
    This function is not being used, but is here for reference that this is another way directly by building an rtree
    index and then performing the filter refine techniques to get the actual geometries that we want. This technique
    is dramatically slower than the earlier function, so as a result is not being used in the notebook file, but
    if called returns the same results, but after a much longer time.
    :param sa2: geopandas geodataframe file containing the data of SA2 areas
    :return: accidents_locations, precise_matches accidents_locations and sa2_areas_of_interest are geopandas
    geodataframes. accidents_locations contains data related to the accident along with the name of the SA2 where the
    accident took place. sa2_areas_of_interest are those SA2 where the accidents may have taken place.
    """
    required_columns = ['AccidentNu', 'DayOfWeek', 'VehicleTyp', 'SevereAcci', 'geometry']
    accidents_locations = read_shape_file('output/AccidentsLocation.shp', required_columns)
    # The SA2 geodataframe has geometries which had null as values, and due to those values we were not able to
    # transform the crs of the geometries, so we are removing the rows with null geometries.
    sa2 = sa2.loc[sa2['geometry'].notnull()]
    sa2 = sa2.to_crs(epsg=4326)

    # I want to get the total bounds of all of the accident locations combined, because then i will be
    # able to ignore those areas from the sa2 shape file which fall out of the bounds
    # reducing the search space. I will then use rtree indexing to further reduce the
    # search space for the accidents locations. This is the filter stage of search.

    # This function takes approximately 5 minutes 30 seconds to complete its execution. The reason for this long
    # execution time is because we have a really large search space where the data frame accidents_locations has
    # over 100000 geometries (rows). In order to further improve the performance we are using the technique of
    # pruning away the geometries in the SA2 geometries that are not intersecting with the accidents_locations
    # geometries.

    bounds = accidents_locations.geometry.total_bounds
    min_x = bounds[0] - 1
    min_y = bounds[1] - 1
    max_x = bounds[2] + 1
    max_y = bounds[3] + 1

    sa2_areas_of_interest = sa2[(sa2.geometry.bounds['minx'] >= min_x) & (sa2.geometry.bounds['maxx'] <= max_x) & (
            sa2.geometry.bounds['miny'] >= min_y) & (sa2.geometry.bounds['maxy'] <= max_y)]

    # sindex = sa2_areas_of_interest.sindex

    # I am building a rtree manually here. I could have used the the above commented line to get the rtree index, but
    # I want to illustrate how to build a rtree index.
    sa2_index = index.Index()
    num = sa2_areas_of_interest.shape[0]
    for i in range(0, num):
        sa2_index.insert(i, sa2_areas_of_interest.iloc[i]['geometry'].bounds)

    sa2_names_list = []
    for i, row in accidents_locations.iterrows():
        # This is doing a filter search where any SA2 geometry which is possibly intersection with the point geometry.
        # At this stage, there may be other geometries which do not really intersect but have been retrieved due to
        # how rtrees are structured. The next refine stage will get us our actual SA2 geometries.
        possible_matches_index = list(sa2_index.intersection(row['geometry'].bounds))
        possible_matches = sa2_areas_of_interest.iloc[possible_matches_index]

        # We are refining our search by checking if the SA2 geometry intersects with our point geometry.
        precise_matches = possible_matches[possible_matches.intersects(row['geometry'])]
        sa2_names_list.append(precise_matches.SA2_NAME16.iloc[0])

    sa2_names = geopandas.GeoSeries(sa2_names_list)
    accidents_locations['SA2'] = sa2_names
    return accidents_locations, sa2_areas_of_interest


def build_bins_labels(data):
    """
    This function builds the bins and corresponding labels to be used in plots
    :param data: geopandas geoseries
    :return: bins, labels
    """
    bins = np.arange(0, data.values.max(), 100)
    labels = []
    bins = bins.astype(int)
    bins = list(bins)
    bins.append(data.values.max() + 1000)

    for i in range(len(bins) - 2):
        bins[i] = int(bins[i])
        if i == 0:
            pass
            labels.append("{}".format(bins[i]))
        else:
            labels.append("{}> and <={}".format(bins[i - 1], bins[i]))

    labels.append("{}>".format(bins[-2]))
    return bins, labels


def calculate_density(x):
    """
    This function calculates the density of the number of accidents by the area of the SA2 area
    :param x: geopandas geodataframe
    :return: float density value
    """
    return x['NUM_ACCIDENTS'] / x['AREASQKM16']


def categorizeTypeOfDay(x):
    """
    This function returns either the day provided is a week day or a weekend.
    :param x: string value of day
    :return: string
    """
    if x == 'Monday':
        return 'Week Day'
    elif x == 'Tuesday':
        return 'Week Day'
    elif x == 'Wednesday':
        return 'Week Day'
    elif x == 'Thursday':
        return 'Week Day'
    elif x == 'Friday':
        return 'Week Day'
    else:
        return 'Weekend'


def spatial_visual_analysis(accidents_locations, sa2_areas_of_interest, html_content):
    """

    :param accidents_locations: geopandas geodataframe containing data related to accident locations
    :param sa2_areas_of_interest: geopandas geodataframe containing data related to the sa2 regions that have been filtered out.
    :param html_content: string containing html content
    :return: html_content, sa2_with_accident_counts
    """
    # I am building the data for getting the number of accidents for each SA2 where an accident has occured.
    num_accidents_sa2 = group_by_data(accidents_locations, ['SA2'])
    num_accidents_sa2['SA2_NAME16'] = num_accidents_sa2['SA2']
    num_accidents_sa2['NUM_ACCIDENTS'] = num_accidents_sa2['count']
    num_accidents_sa2 = num_accidents_sa2.drop(['SA2', 'count'], axis=1)

    sa2_with_accident_counts = pd.merge(sa2_areas_of_interest, num_accidents_sa2,
                                        on=["SA2_NAME16"], how="left")
    sa2_with_accident_counts = sa2_with_accident_counts.loc[sa2_with_accident_counts['NUM_ACCIDENTS'].notnull()]

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.axis('off')
    ax.set_title('Number of accidents in each SA2 for 2006 to 2016')
    sa2_with_accident_counts.plot(column='NUM_ACCIDENTS', ax=ax, edgecolor='k',
                                  scheme="quantiles", k=10, cmap='summer', legend=True, linewidth=0.2)
    fig.savefig('output/num_accidents_sa2.jpg', dpi=500, cmap='summer', edgecolor='black', linewidth=0.2, alpha=1,
                bbox_inches='tight')

    # I am building the data for number of accidents that took place on a week day and a weekend
    accidents_locations['TypeOfDayOfWeek'] = accidents_locations.DayOfWeek.apply(categorizeTypeOfDay)

    num_accidents_by_day_type = group_by_data(accidents_locations, ['TypeOfDayOfWeek', 'SA2'])
    num_accidents_by_day_type['SA2_NAME16'] = num_accidents_by_day_type['SA2']
    num_accidents_by_day_type['NUM_ACCIDENTS'] = num_accidents_by_day_type['count']
    num_accidents_by_day_type = num_accidents_by_day_type.drop(['SA2', 'count'], axis=1)

    sa2_with_week_day_type = pd.merge(sa2_areas_of_interest, num_accidents_by_day_type,
                                      on=["SA2_NAME16"], how="left")

    # sa2_with_accident_counts[['NUM_ACCIDENTS']] = sa2_with_accident_counts[['NUM_ACCIDENTS']].fillna(value=0)
    sa2_with_week_day_type = sa2_with_week_day_type.loc[sa2_with_week_day_type['NUM_ACCIDENTS'].notnull()]
    sa2_with_week_day_type_weekend = sa2_with_week_day_type[sa2_with_week_day_type['TypeOfDayOfWeek'] == 'Weekend']
    sa2_with_week_day_type_weekday = sa2_with_week_day_type[sa2_with_week_day_type['TypeOfDayOfWeek'] == 'Week Day']

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.axis('off')
    ax.set_title('Number of accidents in each SA2 for 2006 to 2016 during the weekdays')
    sa2_with_week_day_type_weekday.plot(column='NUM_ACCIDENTS', ax=ax, edgecolor='k',
                                        scheme="quantiles", k=10, cmap='summer', legend=True, linewidth=0.2)
    fig.savefig('output/num_accidents_sa2_weekday.jpg', dpi=500, cmap='summer', edgecolor='black', linewidth=0.2,
                alpha=1, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.axis('off')
    ax.set_title('Number of accidents in each SA2 for 2006 to 2016 during the weekends (Saturday and Sunday)')
    sa2_with_week_day_type_weekend.plot(column='NUM_ACCIDENTS', ax=ax, edgecolor='k',
                                        scheme="quantiles", k=10, cmap='summer', legend=True, linewidth=0.2)
    fig.savefig('output/num_accidents_sa2_weekend.jpg', dpi=500, cmap='summer', edgecolor='black', linewidth=0.2,
                alpha=1, bbox_inches='tight')

    image_link = """
    <div class='row'>
        <div class='col-md-12'>
            <img src='{0}' class='img-fluid'/>
            <br />
            <center>
                <label>
                    Figure 1: The figure shows the number of accidents that took place from 2006 to 2017 during both 
                    the weekdays and the weekends. Do note the bins in the legends.
                </label>
            </center>

        </div>
    </div>
    <br />
    <div class='row'>
        <div class='col-md-12'>
            <img src='{1}' class='img-fluid'/>
            <br />
            <center>
                <label>
                    Figure 2: The figure shows the number of accidents that took place from 2006 to 2017 during the weekdays. 
                    Do note the bins in the legends. The maximum number of accident is less than what we have in Figure 1.
                </label>
            </center>
        </div>
    </div>    
    <div class='row'>    
        <div class='col-md-12'>
            <img src='{2}' class='img-fluid'/>
            <center>
                <label>
                    Figure 3: The figure shows the number of accidents that took place from 2006 to 2017 during the weekend. 
                    Do note the bins in the legends. The maximum number of accident is less than what we have in Figure 1.
                    The bins tell that there were alot more fewer accidents on the weekends than on the weekdays and also as an
                    overall total.
                </label>
            </center>
        </div>
    </div>

    """.format('num_accidents_sa2.jpg', 'num_accidents_sa2_weekday.jpg', 'num_accidents_sa2_weekend.jpg')

    text = """
    <p>
        The first visualization shows the number of accidents that took place in the shown SA2 regions from 2006 to 2016. 
        This is for all days (weekday and weekend) when the accidents took place.
        The second visualization shows the number of accidents that took place in the shown SA2 regions from 2006 to 2016 during 
        the weekdays (Monday to Friday).
        The third visualization shows the the number of accidents that took place in the shown SA2 regions from 2006 to 2016
        during the weekends (Saturday and Sunday).
        The legend shows the bin ranges of the number of accidents that took place.
    </p>    
    """
    html_content += build_html_component('', 'Spatial Temporal Visual Analysis (Basic)')

    html_content += build_html_component_without_title(text)

    html_content += build_html_component_with_html(image_link)
    write_to_file(html_content, 'task3_922939.html', 'Task 3 922939')
    return html_content, sa2_with_accident_counts


def spatial_autocorrelation(sa2_with_accident_counts, html_content):
    """
    This function performs spatial autocorrelation and calculates whether the data is spatially correlated or not.
    :param sa2_with_accident_counts: geopandas geodataframe containing data related to the sa2 areas with accident counts.
    :param html_content: string containing html content
    :return: html_content, sa2_with_accident_counts
    """
    # I am building weights index based on the Queen Contiguity which is a neighbor finding algorithm. If an sa2 area is
    # a neighbor of the sa2 area in question, then a 1 will be set for that neighbor else 0.
    weights_queen_contiguity = lps.weights.Queen.from_dataframe(sa2_with_accident_counts)
    # The weights are being transformed into a row standardization method. Each value in the row of
    # the spatial weights matrix is rescaled so that their
    # sum equals to 1.
    weights_queen_contiguity.transform = 'r'
    non_spatial_attribute = sa2_with_accident_counts.NUM_ACCIDENTS.values
    # along with the weights, i also have to give a non-spatial attribute in order to perform autocorrelation,
    # and since i want see if neighboring
    # sa2 areas have similar numbers of accidents or not.
    non_spatial_attribute_lag = lps.weights.lag_spatial(weights_queen_contiguity, non_spatial_attribute)
    # non_spatial_attribute_lag_5_buckets = mc.Quantiles(non_spatial_attribute_lag, k=5)
    sa2_with_accident_counts['LAG_NUM_ACCIDENTS'] = non_spatial_attribute_lag
    # The spatial lag that has been calculated tells how similar one spatial area is similar to its neighbors.
    # Essentially we want to find if there are clusters SA2 areas where there are a high number of accidents.

    fig, ax = plt.subplots(figsize=(20, 20))
    sa2_with_accident_counts.plot(column='LAG_NUM_ACCIDENTS', ax=ax, edgecolor='k',
                                  scheme='quantiles', cmap='summer', k=10, legend=True, linewidth=0.2)
    ax.set_title("Spatial Lag Number of Accidents from 2006 to 2016")
    ax.axis('off')
    fig.savefig('output/spatial_lag_num_accidents_quantiles.jpg', dpi=200, cmap='summer', edgecolor='black',
                linewidth=0.2, alpha=1, bbox_inches='tight')

    # The null hypothesis that we take is that the clusters are spatially randomly distributed.
    # We will be using the approach of Permutation Inference where Pysal will build randomized reference
    # distributions (999 is by default). We will then calculate a pseudo p-value from the randomized distributions
    # and compare with our observed (actual) results that we had obtained from our model.
    # A really brief overview on Joint Count Statistics has been given by Luc Anselin on
    # Youtube https://www.youtube.com/watch?time_continue=40&v=BdsdYEbUkj4&t=1071s
    # Pysal also has provided with a really good easy to follow step by step on how to perform
    # Spatial Autocorrelation Analysis https://nbviewer.jupyter.org/github/pysal/esda/blob/master/notebooks/Spatial%20Autocorrelation%20for%20Areal%20Unit%20Data.ipynb

    non_spatial_attribute_lag_binary = non_spatial_attribute_lag > np.median(non_spatial_attribute_lag)
    sum(non_spatial_attribute_lag_binary)  # There are 230 sa2 areas which are above the median value of 271.0
    num_true = 0
    num_false = 0
    # sa2 areas that are higher than the median will be assigned 1 High, while the rest will be assigned 0 Low
    non_spatial_attribute_lag_binary_labeled = []
    labels = ["0 Low", "1 High"]

    for i in non_spatial_attribute_lag_binary:
        if i == True:
            num_true = num_true + 1
            non_spatial_attribute_lag_binary_labeled.append(labels[1])
        else:
            num_false = num_false + 1
            non_spatial_attribute_lag_binary_labeled.append(labels[0])

    sa2_with_accident_counts['YB_LABELED'] = non_spatial_attribute_lag_binary_labeled
    fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'aspect': 'equal'})
    sa2_with_accident_counts.plot(column='YB_LABELED', cmap='binary', edgecolor='grey', legend=True, ax=ax)
    ax.set_title("High number of accidents and low number of accidents segmentation from 2006 to 2016")
    ax.axis('off')
    fig.savefig('output/high_low_segments_num_accidents.jpg', dpi=500, cmap='summer', edgecolor='black', linewidth=0.2,
                alpha=1, bbox_inches='tight')

    # We want to find our joint count statistics, so we need the data in a binary form
    # In addition, Pysal will build randomized distributions to compare our observed (actual) results against.
    non_spatial_attribute_lag_binary = 1 * (non_spatial_attribute_lag > np.median(non_spatial_attribute_lag))
    weights_queen_contiguity = lps.weights.Queen.from_dataframe(sa2_with_accident_counts)
    weights_queen_contiguity.transform = 'b'
    np.random.seed(12345)
    joint_counts = esda.join_counts.Join_Counts(non_spatial_attribute_lag_binary, weights_queen_contiguity)
    '''
    print(joint_counts.bb)# number of black-black joins
    print(joint_counts.ww)# number of white-white joins
    print(joint_counts.bw)# number of black-white or white-black joins
    print((joint_counts.bb+joint_counts.ww+joint_counts.bw))
    # I have divided by two to remove the double counting. S0 contains the count of the join, which is a 2-way count,
    # but i only want 1-way, so have to divide by 2.
    print(weights_queen_contiguity.s0/2) 

    joint_counts.mean_bb # This is the mean of black-black joint count from the randomized distribution
    # We have observed that the mean value is lower than our actual black-black joins, but we have to 
    #find out how far off from the mean is our actual result really.
    '''

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(
        'Density plot of randomized distributions number of black-black joins (joint count) and the actual value')
    sbn.kdeplot(joint_counts.sim_bb, shade=True, ax=ax)
    plt.vlines(joint_counts.bb, 0, 0.075, color='r')
    plt.vlines(joint_counts.mean_bb, 0, 0.075)
    plt.xlabel('Number of Black-Black Joins')
    fig.savefig('output/kdeplot.jpg', dpi=100, bbox_inches='tight')

    # The density distribution plot of the randomized runs shows that our actual result (red line)
    # is an extreme value and is not within the distribution we can reject the null hypothesis that
    # the accident numbers in the sa2 areas are spatially random, when infact they are autocorrelated.
    # The pseudo p-value also supports our claim with a result of 0.001.

    image_link = """
    <div class='row'>
        <div class='col-md-12'>
            <img src='{0}' class='img-fluid'/>
            <center>
                <label>
                    Figure 4: This figure shows the spatial lag that has been calculated using Queen contiguity statistic which 
                    builds an index where if an area is a neighbor is given a value of 1 and 0 if not. Then the spatial lag is calculated
                    which tells how similar an area is similar to its neighbors.
                </label>
            </center>
        </div>
    </div>
    <div class='row'>
        <div class='col-md-12'>
            <img src='{1}' class='img-fluid'/>
            <center>
                <label>
                    Figure 5: This figure shows the SA2 areas being categorized as being higher than the median value of the spatial lag (black in color)
                    and lower than the spatial lag (white in color). We are regarding the black areas as having high number of accidents, while the 
                    white areas as having low number of accidents. We can already see that there are clusters grouped together. We will statistically
                    prove this in the next figure.
                </label>
            </center>
        </div>
    </div>    
    <div class='row'>    
        <div class='col-md-12'>
            <center><img src='{2}' class='img-fluid'/></center>
            <center>
                <label>
                    Figure 6: The figure shows the results of the statisticall p-value test that we have done in order to invalidate the null hypothesis.
                    We have used Pysal, and it generates by default 999 randomized  clusters and then finds their mean and other values. The density plot 
                    shown in the figure is of the randomized generations, and the black line shows the mean. However, our own actual value is an extreme 
                    value and this proves that the Null-hypotheses is wrong and that we can reject it, therefore it can be said that our data contains
                    clusters that are closely grouped together, leading to the fact that those SA2 areas are similar to each other.
                </label>
            </center>
        </div>
    </div>

    """.format('spatial_lag_num_accidents_quantiles.jpg', 'high_low_segments_num_accidents.jpg', 'kdeplot.jpg')

    text = """
    <p>
        This section of the project required me to perform autocorrelation on the spatial data. What this means was that I was to find how similar the
        SA2 areas are to their neighbors and whether there are any clustering or not. Infact this is what autocorrelation tells me. To begin with 
        we take a Null-hypothesis that the clusters are randomly distributed and there are no groups clustered together. We will show that this 
        hypothesis is false, and that in reality there are clusters which are grouped together and are not randomized.
    </p>    
    """
    html_content += build_html_component('', 'Spatial Autocorrelation Calculation (Advanced')
    html_content += build_html_component_without_title(text)
    html_content += build_html_component_with_html(image_link)
    write_to_file(html_content, 'task3_922939.html', 'Task 3 922939')
    return html_content, sa2_with_accident_counts


def plot(X, model):
    """
    This code plots the clustered points. This function has been inspired from the
    DBSCAN example (https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html) and the example provide
    by Elham https://notebooks.azure.com/ElhamN/projects/GEOM90042-1/html/L6/plotClusters.py
    :param X: a 2D numpy array of coordinates
    :param model: a model containg the DBSCAN result
    """
    # plotting the results
    labels = model.labels_
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)
        plt.savefig('output/dbscan_cluster.jpg', dpi=200, bbox_inches='tight')


def classify_helmet_belt_worn(x):
    """
    This function returns a strinig representation of the int value of the field which specifies whether the
    person was wearing a setabelt or a helmet. This specification is from the Road Crash Statistics Victoria , 2013 Edition
    document.
    :param x: int value representing the classify helmet belt worn field
    :return: string representation of the integer value
    """
    if x == 1:
        return 'Seatbelt Worn'
    elif x == 2:
        return 'Seatbelt Not Worn'
    elif x == 3:
        return 'Child Restraint Worn'
    elif x == 4:
        return 'Child Restraint Not Worn'
    elif x == 5:
        return 'Seatbelt/restraint Not fitted'
    elif x == 6:
        return 'Crash Helmet Worn'
    elif x == 7:
        return 'Crash Helmet Not Worn'
    elif x == 8:
        return 'Not Appropriate'
    else:
        return 'Not Known'


def clustering_analysis(accidents_location_data_frame, road_surface_data_frame, atomospheric_con_data_frame,
                        person_data_frame, accidents_locations, sa2_with_accident_counts, html_content):
    """
    This function prepares data for analysis via DBSCAN and then brings together other data and builds them into
    tables for further analysis.

    :param accidents_location_data_frame: geopandas geodataframe containing accidents location
    :param road_surface_data_frame: pandas dataframe containing data related to the road surface condition
    :param atomospheric_con_data_frame: pandas dataframe containing data related to the atmospheric condition
    :param person_data_frame: pandas dataframe containing data related to people involved in the accident
    :param accidents_locations: geopandas geodataframe containing data related to accidents
    :param sa2_with_accident_counts: geopandas geodataframe of sa2 areas
    :param html_content: string containing html data
    :return: group_by_num_parties, group_by_atmos_surface, m, group_by_age_group, group_by_sex, group_by_helmet_worn, coords, db, html_content
    """
    # I want to study the sa2 areas which were marked as having a high number of accidents
    sa2_with_accident_counts_with_clustered = sa2_with_accident_counts[sa2_with_accident_counts.YB_LABELED == '1 High']
    # I am interested in the sa2 area which has the highest number of accidents from those areas
    # that have been categorized as high
    max_num_accidents = sa2_with_accident_counts_with_clustered.NUM_ACCIDENTS.max()
    sa2_with_max_num_accidents = sa2_with_accident_counts_with_clustered[
        sa2_with_accident_counts_with_clustered.NUM_ACCIDENTS == max_num_accidents]
    accidents_from_max_accidents_sa2 = geopandas.sjoin(accidents_locations, sa2_with_max_num_accidents, how='inner',
                                                       op='within')

    accidents_from_max_accidents_sa2['LATITUDE'] = accidents_from_max_accidents_sa2.geometry.apply(lambda x: x.y)
    accidents_from_max_accidents_sa2['LONGITUDE'] = accidents_from_max_accidents_sa2.geometry.apply(lambda x: x.x)
    accidents_locations_to_be_clustered = get_columns_from_data(accidents_from_max_accidents_sa2,
                                                                ['TypeOfDayOfWeek', 'LATITUDE', 'LONGITUDE'])
    accidents_locations_to_be_clustered.LATITUDE = accidents_locations_to_be_clustered.LATITUDE.astype(float)
    accidents_locations_to_be_clustered.LONGITUDE = accidents_locations_to_be_clustered.LONGITUDE.astype(float)
    coords = np.vstack((accidents_locations_to_be_clustered[['LATITUDE', 'LONGITUDE']]['LATITUDE'].values,
                        accidents_locations_to_be_clustered[['LATITUDE', 'LONGITUDE']]['LONGITUDE'].values)).T

    # define the number of kilometers in one radian
    kms_per_radian = 6371.0088
    epsilon = 0.05 / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=3, metric='haversine').fit(np.radians(coords))
    '''cluster_labels = db.labels_
    # get the number of clusters
    num_clusters = len(set(cluster_labels))
    # set(cluster_labels)
    num_clusters'''

    # The results of the DBSCAN gives us 29 clusters. Since we are clustering on the longitude and
    # latitude the accident locations that are spatially closer together are clustered together.
    # plot(coords, db)

    # In order to further understand what may have caused the accidents, I will be looking at a
    # number of variables such as the road condition, atmospheric conditions, information about the drivers, etc...

    # I want to explore the condition of the road at the time of the accident
    # Dry, Wet, Muddy, Snowy, Icy, Unknown
    road_surface_data_frame.rename(index=str,
                                   columns={'ACCIDENT_NO': 'AccidentNu', 'Surface Cond Desc': 'SurfaceCondDesc'},
                                   inplace=True)
    road_surface_data_frame = get_columns_from_data(road_surface_data_frame, ['AccidentNu', 'SurfaceCondDesc'])

    # I want to explore the atmospheric condition at the time of the accident
    # Clear, Raining, Snowing, Fog, Smoke, Dust, Strong Winds, Unknown
    atomospheric_con_data_frame.rename(index=str,
                                       columns={'ACCIDENT_NO': 'AccidentNu', 'Atmosph Cond Desc': 'AtmosCondDesc'},
                                       inplace=True)
    atomospheric_con_data_frame = get_columns_from_data(atomospheric_con_data_frame, ['AccidentNu', 'AtmosCondDesc'])

    accidents_from_max_accidents_sa2 = pd.merge(accidents_from_max_accidents_sa2, atomospheric_con_data_frame,
                                                on="AccidentNu",
                                                how="inner")
    accidents_from_max_accidents_sa2 = pd.merge(accidents_from_max_accidents_sa2, road_surface_data_frame,
                                                on="AccidentNu",
                                                how="inner")

    # suburb = accidents_from_max_accidents_sa2.iloc[0].SA3_NAME16
    # greater = accidents_from_max_accidents_sa2.iloc[0].GCC_NAME16
    # city = accidents_from_max_accidents_sa2.iloc[0].SA2_NAME16
    # state = accidents_from_max_accidents_sa2.iloc[0].STE_NAME16
    # print("{}, {}, {}, {}".format(suburb, greater, city, state))

    m = folium.Map(
        location=[coords[0][0],
                  coords[0][1]],
        zoom_start=15)

    for coord in coords:
        folium.CircleMarker(
            location=[coord[0], coord[1]],
            radius=5,
            color='#ff0000',
            fill=True,
            fill_color='#ff0000'
        ).add_to(m)
    m.save('output/folium_map_cluster.html')
    folium_map = m._repr_html_()


    accidents_location_data_frame.rename(index=str, columns={'ACCIDENT_NO': 'AccidentNu'}, inplace=True)
    accidents_location_data_frame.head(3)
    accidents_from_max_accidents_sa2 = pd.merge(accidents_from_max_accidents_sa2, accidents_location_data_frame,
                                                on="AccidentNu",
                                                how="inner")

    accidents_from_max_accidents_sa2['NUM_PARTIES_INVOLVED'] = accidents_from_max_accidents_sa2.VehicleTyp.apply(
        lambda x: len(x.split(',')))

    number_parties_involved_html = group_by_data(accidents_from_max_accidents_sa2, ['NUM_PARTIES_INVOLVED']).to_html()
    group_by_num_parties = group_by_data(accidents_from_max_accidents_sa2, ['NUM_PARTIES_INVOLVED'])
    group_by_atmos_surface_html = group_by_data(accidents_from_max_accidents_sa2,
                                                ['AtmosCondDesc', 'SurfaceCondDesc']).to_html()
    group_by_atmos_surface = group_by_data(accidents_from_max_accidents_sa2, ['AtmosCondDesc', 'SurfaceCondDesc'])
    group_by_atmos_cond = group_by_data(accidents_from_max_accidents_sa2, ['AtmosCondDesc'])
    group_by_road_surface = group_by_data(accidents_from_max_accidents_sa2, ['SurfaceCondDesc'])

    # The majority of the accidents have been when the atmospheric conditions were clear and the road
    # surface condition was dry, so there  must be some other reason as to why there were
    # so many accidents. I will now examine the drivers who were involved in the accidents, and
    # whether we can find any anomalies there.
    person_data_frame.rename(index=str, columns={'ACCIDENT_NO': 'AccidentNu'}, inplace=True)
    accidents_from_max_accidents_sa2 = pd.merge(accidents_from_max_accidents_sa2, person_data_frame,
                                                on="AccidentNu",
                                                how="inner")

    # group_by_data(person_data_frame, ['AtmosCondDesc','SurfaceCondDesc'])
    accidents_from_max_accidents_sa2_drivers = accidents_from_max_accidents_sa2[
        accidents_from_max_accidents_sa2['Road User Type Desc'] == 'Drivers']

    age_group_html = group_by_data(accidents_from_max_accidents_sa2_drivers, ['Age Group']).to_html()
    group_by_age_group = group_by_data(accidents_from_max_accidents_sa2_drivers, ['Age Group'])
    average_driver_age = round(accidents_from_max_accidents_sa2_drivers.AGE.mean(), 2)

    group_by_sex_html = group_by_data(accidents_from_max_accidents_sa2_drivers, ['SEX']).to_html()
    group_by_sex = group_by_data(accidents_from_max_accidents_sa2_drivers, ['SEX'])

    accidents_from_max_accidents_sa2_drivers[
        'HELMET_BELT_WORN_CAT'] = accidents_from_max_accidents_sa2_drivers.HELMET_BELT_WORN.apply(
        classify_helmet_belt_worn)
    helmet_belt_worn_html = group_by_data(accidents_from_max_accidents_sa2_drivers, ['HELMET_BELT_WORN_CAT']).to_html()
    group_by_helmet_worn = group_by_data(accidents_from_max_accidents_sa2_drivers, ['HELMET_BELT_WORN_CAT'])

    image_link = """
    <div class='row'>
        <div class='col-md-12'>
            <img src='{0}' class='img-fluid'/>
            <center>
                <label>
                    Figure 4: This figure shows the spatial lag that has been calculated using Queen contiguity statistic which 
                    builds an index where if an area is a neighbor is given a value of 1 and 0 if not. Then the spatial lag is calculated
                    which tells how similar an area is similar to its neighbors.
                </label>
            </center>
        </div>
    </div>
    <br /><br />
    <div class='row'>    
        <div class='col-md-12'>

            <div class='row'>
                <div class='col-md-12'>
                <p>
                The Folium map shows where the clusters are exactly located further helping us in analyzing the data. We can see that 
                there is a high concentration of accidents on La Trobe Street so will be further examining the data for it.
                We decided to analyze this particular area because it contains the highest number of accidents in all of the SA2 areas.
                </p>
                </div>
            </div>

            <div class='row'>
                <div class='col-md-12'>
                {1}
                </div>
            </div>

        </div>
    </div>    

    <br /><br />
    <div class='row'>    
        <div class='col-md-12'>

            <div class='row'>
                <div class='col-md-12'>
                <p>
                    These table further give a better understanding of our accidents data that took place on La Trobe
                    Street.
                </p>
                </div>
            </div>

            <div class='row'>
                <div class='col-md-4'>
                <center>
                    {2}
                    <br />
                    Table 1: The table shows the number of parties that were involved in the accident. This further tells us how serious the accident was
                    due to the number of parties that were involved.
                </center>

                </div>
                <div class='col-md-4'>
                <center>
                    {3}
                    <br />
                    Table 2: The table shows the results of what the atmospheric and road conditions were at the time of the accident. We can see
                    that the majority of the severe accidents took place when the weather was clear and the road surface condition was dry.
                    There must be some other reason as to why the accidents took place.
                </center>
                </div>
                <div class='col-md-4'>
                <center>
                    {4}
                    <br />
                    Table 3: The table shows the number of drivers and their age groups who were involved in the accidents. 
                    The average age of the drivers was {7}.
                 </center>
                </div>
            </div>

        </div>
    </div>    

    <br /><br />
    <div class='row'>    
        <div class='col-md-12'>

            <div class='row'>
                <div class='col-md-6'>
                <center>
                {5}
                <br />
                Table 4: The table shows the gender statistics of drivers.
                </center>
                </div>

                <div class='col-md-6'>
                <center>
                {6}
                Table 5: The table shows the how many people wore their seatbelts/helmets and how many didnt.
                </center>
                </div>
            </div>

        </div>
    </div>

    """.format('dbscan_cluster.jpg', folium_map, number_parties_involved_html, group_by_atmos_surface_html,
               age_group_html, group_by_sex_html, helmet_belt_worn_html, average_driver_age)

    text = """
    <p>
        The results of the DBSCAN gives us 29 clusters. Since we are clustering on the longitude and latitude the accident locations that are
        spatially closer together are clustered together.

        Once we have the clusters, we had to further analyze what the reason may be why there were so many accidents. 
        In order to conduct this I looked at a number of variables such as the road condition, atmospheric conditions, information
        about the drivers, etc...
    </p>    
    """
    html_content += build_html_component('', 'Clustering Analysis (More Advanced)')

    html_content += build_html_component_without_title(text)

    html_content += build_html_component_with_html(image_link)
    return group_by_num_parties, group_by_atmos_surface, m, group_by_age_group, group_by_sex, group_by_helmet_worn, coords, db, accidents_from_max_accidents_sa2, group_by_atmos_cond, group_by_road_surface, html_content


def plot_scatter_plot(x, y, x_axis_label, y_axis_label, title):
    """
    This function plots a scatter plot of values of x against the y
    :param x: list of x values
    :param y: list of y values
    :param x_axis_label: list of labels for x
    :param y_axis_label: list of labels for y
    :param title: string value for the title of the plot
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(x_axis_label, fontsize=10)
    ax.set_ylabel(y_axis_label, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=10, rotation=30)
    ax.set_title(title)
    # ax.legend(loc='upper right', frameon=True)
    plt.show()
    file_name = ((((title + '.jpg').replace('/', ''))).replace(' ', '_')).lower()
    fig.savefig('output/' + file_name, dpi=100, bbox_inches='tight')


def plot_bar_plot(x, y, x_axis_label, y_axis_label, title):
    """
    This function plots a bar plot of values of x against the y
    :param x: list of x values
    :param y: list of y values
    :param x_axis_label: list of labels for x
    :param y_axis_label: list of labels for y
    :param title: string value for the title of the plot
    """
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xlabel(x_axis_label, fontsize=10)
    ax.set_ylabel(y_axis_label, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=10, rotation=90)
    ax.set_title(title)
    # ax.legend(loc='upper right', frameon=True)
    plt.show()
    file_name = ((((title + '.jpg').replace('/', ''))).replace(' ', '_')).lower()
    fig.savefig('output/' + file_name, dpi=100, bbox_inches='tight')


def calculate_correlation_coefficient(x, y):
    """
    This function calculates the correlation coefficient of between the x and y values
    :param x: list of x values
    :param y: list of y values
    :return: correlation coefficient
    """
    numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    denominator = np.sqrt(np.sum((x - np.mean(x)) ** 2) * np.sum((y - np.mean(y)) ** 2))
    if denominator > 0:
        return numerator / denominator
    else:
        return 0


def getTypeOfHourDay(x):
    """
    This function returns a string representation of the type of hour of day
    :param x: int value of hour
    :return: string representation of the type of hour of day
    """
    hour = int(x.split('.')[0])
    if hour >= 0 and hour <= 5:
        return 'Early Morning'
    elif hour >= 6 and hour <= 11:
        return 'Morning'
    elif hour >= 12 and hour <= 14:
        return 'Early Afternoon'
    elif hour >= 15 and hour <= 17:
        return 'Afternoon'
    elif hour >= 18 and hour <= 19:
        return 'Early Evening'
    elif hour >= 20 and hour <= 23:
        return 'Night'
