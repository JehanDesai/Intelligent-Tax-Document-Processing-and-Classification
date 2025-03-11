import mysql.connector

class Models:
    __connection = None
    __cursor = None
    __username = "root"
    __password = "garethbale11"
    def __init__(self):
        self.__connection = mysql.connector.connect(host = "localhost", username = self.__username, password = self.__password)
        if self.__connection.is_connected():
            print("Connection established....")
            self.__cursor = self.__connection.cursor()

    def createDatabase(self):
        query = "create database if not exists assignment"
        print("Database created..." if self.__cursor.execute(query) else "ERROR!!!")

    def useDatabase(self):
        self.__connection = mysql.connector.connect(
            host = "localhost",
            username = self.__username,
            password = self.__password,
            database = "assignment"
        )
        self.__cursor = self.__connection.cursor()
        print("Using database assignment...")
    
    def createTable(self):
        query = "create table if not exists tax_file_storage(id int AUTO_INCREMENT PRIMARY KEY, projectCode varchar(50) not null, uploaded_tax_file LONGBLOB NOT NULL, processed_tax_file LONGBLOB NOT NULL, uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        print("ERROR!!!" if  self.__cursor.execute(query) else "Table created...")
    
    def insertIntoTable(self):
        query = "insert into tax_file_storage(projectCode, uploaded_tax_file, processed_tax_file) values(%s, %s, %s)"
        with open("temp.txt", "r") as file:
            text = file.read()
        self.__cursor.execute(query, ("Project-Honeywell", text, text))
        self.__connection.commit()
        print("File successfully stored!!!")
    
    def displayAll(self):
        query = "select * from tax_file_storage"
        records = self.__cursor.fetchall()
        for rows in records:
            print("id:", rows[0])
            print("project code:", rows[1])
            print("Uploaded tax file content:", rows[2])
            print("Processed tax file content:", rows[3])
            print("Uploaded on:", rows[4])
        print()

    def deleteFromTheTable(self, projectCode, uploadedAt):
        query = "delete from tax_file_storage where projectCode = %s and uploaded_at = %s"
        done = self.__cursor.execute(query, (projectCode, uploadedAt))
        if done:
            self.__connection.commit()
            print("Successfully removed the record...")
        else:
            print("ERROR!!!")
    
    