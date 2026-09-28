import mysql.connector


def get_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="2007",
        database="deepfake_detection"
    )

    return connection


if __name__ == "__main__":

    try:
        connection = get_connection()

        if connection.is_connected():
            print("MySQL connection successful!")

        connection.close()

    except mysql.connector.Error as error:
        print("MySQL connection failed!")
        print("Error:", error)