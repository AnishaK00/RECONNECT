import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import 'package:crypto/crypto.dart';
import 'dart:convert';
import 'dart:io';
import 'patient_model.dart';
import 'emergencycon_model.dart';
import 'caregiver_model.dart';
import 'enrolled_person.dart';

class DatabaseHelper {
  static final DatabaseHelper _instance = DatabaseHelper._internal();
  factory DatabaseHelper() => _instance;
  DatabaseHelper._internal();

  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final path = join(await getDatabasesPath(), 'reconnect.db');
    return openDatabase(
      path,
      version: 4,
      onConfigure: (db) async {
        await db.execute('PRAGMA foreign_keys = ON');
      },
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE caregivers(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            phone TEXT NOT NULL UNIQUE,
            passwordHash TEXT NOT NULL,
            caregiverType TEXT NOT NULL,
            familyRelation TEXT,
            profession TEXT
          )
        ''');

        await db.execute('''
          CREATE TABLE patients(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullName TEXT NOT NULL,
            patientId TEXT NOT NULL UNIQUE,
            username TEXT NOT NULL UNIQUE,
            passwordHash TEXT NOT NULL,
            dob TEXT NOT NULL,
            phone TEXT NOT NULL,
            email TEXT NOT NULL,
            address TEXT NOT NULL,
            caregiverName TEXT NOT NULL,
            caregiverPhone TEXT NOT NULL,
            medicalInformation TEXT NOT NULL,
            caregiverId INTEGER NOT NULL,
            FOREIGN KEY (caregiverId) REFERENCES caregivers (id)
              ON DELETE CASCADE
          )
        ''');

        await db.execute('''
          CREATE TABLE emergency_contacts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patientId INTEGER NOT NULL,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            relationship TEXT NOT NULL,
            FOREIGN KEY (patientId) REFERENCES patients (id)
              ON DELETE CASCADE
          )
        ''');

        await db.execute('''
          CREATE TABLE enrolled_persons(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patientId INTEGER NOT NULL,
            name TEXT NOT NULL,
            relation TEXT NOT NULL,
            description TEXT NOT NULL,
            photoCount INTEGER NOT NULL,
            FOREIGN KEY (patientId) REFERENCES patients (id)
              ON DELETE CASCADE
          )
        ''');

        await db.execute('''
          CREATE TABLE enrolled_person_photos(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enrolledPersonId INTEGER NOT NULL,
            imageData BLOB NOT NULL,
            FOREIGN KEY (enrolledPersonId) REFERENCES enrolled_persons (id)
              ON DELETE CASCADE
          )
        ''');

        //  create the memory_events table here once the
        // Memory Timeline feature is wired up for real. See the commented
        // schema + methods at the bottom of this file.
      },
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          await db.execute('ALTER TABLE patients ADD COLUMN username TEXT');
        }
        if (oldVersion < 3) {
          await db.execute('''
            CREATE TABLE enrolled_person_photos(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              enrolledPersonId INTEGER NOT NULL,
              imageData BLOB NOT NULL,
              FOREIGN KEY (enrolledPersonId) REFERENCES enrolled_persons (id)
                ON DELETE CASCADE
            )
          ''');
        }
        if (oldVersion < 4) {
          await db.execute(
            "ALTER TABLE patients ADD COLUMN email TEXT NOT NULL DEFAULT ''",
          );
        }
        //  bump version to 5 and add an
        // `if (oldVersion < 5) { ... CREATE TABLE memory_events ... }`
        // block here when the Memory Timeline feature is implemented.
      },
    );
  }

  String hashPassword(String password) {
    return sha256.convert(utf8.encode(password)).toString();
  }

  // ---------- Caregiver methods ----------

  Future<int> insertCaregiver(Caregiver caregiver) async {
    final db = await database;
    return db.insert('caregivers', caregiver.toMap());
  }

  Future<Caregiver?> getCaregiverByEmail(String email) async {
    final db = await database;
    final results = await db.query(
      'caregivers',
      where: 'email = ?',
      whereArgs: [email],
    );
    if (results.isEmpty) return null;
    return Caregiver.fromMap(results.first);
  }

  Future<Caregiver?> getCaregiverByPhone(String phone) async {
    final db = await database;
    final results = await db.query(
      'caregivers',
      where: 'phone = ?',
      whereArgs: [phone],
    );
    if (results.isEmpty) return null;
    return Caregiver.fromMap(results.first);
  }

  Future<bool> emailExists(String email) async {
    return await getCaregiverByEmail(email) != null;
  }

  Future<bool> validateCaregiverLogin(String email, String password) async {
    final caregiver = await getCaregiverByEmail(email);
    if (caregiver == null) return false;
    return caregiver.passwordHash == hashPassword(password);
  }

  // ---------- Patient methods ----------

  Future<Map<String, dynamic>> insertPatientWithContacts(
    Patient patient,
    List<EmergencyContact> contacts,
  ) async {
    final db = await database;
    return db.transaction((txn) async {
      final patientMap = patient.toMap()..remove('id');
      final newId = await txn.insert('patients', patientMap);

      final generatedPatientId = 'PT${newId.toString().padLeft(4, '0')}';
      await txn.update(
        'patients',
        {'patientId': generatedPatientId},
        where: 'id = ?',
        whereArgs: [newId],
      );

      for (final contact in contacts) {
        await txn.insert(
          'emergency_contacts',
          contact.toMap()..['patientId'] = newId,
        );
      }

      return {'id': newId, 'patientId': generatedPatientId};
    });
  }

  Future<Patient?> getPatientByPatientId(String patientId) async {
    final db = await database;
    final results = await db.query(
      'patients',
      where: 'patientId = ?',
      whereArgs: [patientId],
    );
    if (results.isEmpty) return null;
    return Patient.fromMap(results.first);
  }

  Future<Patient?> getPatientByUsername(String username) async {
    final db = await database;
    final results = await db.query(
      'patients',
      where: 'username = ?',
      whereArgs: [username],
    );
    if (results.isEmpty) return null;
    return Patient.fromMap(results.first);
  }

  Future<Patient?> getPatientByEmail(String email) async {
    final db = await database;
    final results = await db.query(
      'patients',
      where: 'email = ?',
      whereArgs: [email],
    );
    if (results.isEmpty) return null;
    return Patient.fromMap(results.first);
  }

  Future<bool> patientIdExists(String patientId) async {
    return await getPatientByPatientId(patientId) != null;
  }

  Future<bool> usernameExists(String username) async {
    return await getPatientByUsername(username) != null;
  }

  Future<bool> patientEmailExists(String email) async {
    return await getPatientByEmail(email) != null;
  }

  Future<bool> validatePatientLogin(String username, String password) async {
    final patient = await getPatientByUsername(username);
    if (patient == null) return false;
    return patient.passwordHash == hashPassword(password);
  }

  Future<List<EmergencyContact>> getContactsForPatient(int patientId) async {
    final db = await database;
    final results = await db.query(
      'emergency_contacts',
      where: 'patientId = ?',
      whereArgs: [patientId],
    );
    return results.map((row) => EmergencyContact.fromMap(row)).toList();
  }

  Future<List<Patient>> getPatientsForCaregiver(int caregiverId) async {
    final db = await database;
    final results = await db.query(
      'patients',
      where: 'caregiverId = ?',
      whereArgs: [caregiverId],
    );
    return results.map((row) => Patient.fromMap(row)).toList();
  }

  // ---------- EnrolledPerson methods ----------

  Future<int> insertEnrolledPerson(EnrolledPerson person) async {
    final db = await database;
    return db.insert('enrolled_persons', person.toMap());
  }

  /// Stores the enrollment and the full image bytes locally in SQLite.
  Future<int> insertEnrolledPersonWithPhotos(
    EnrolledPerson person,
    List<File> photos,
  ) async {
    final db = await database;
    return db.transaction((txn) async {
      final personId = await txn.insert('enrolled_persons', person.toMap());

      for (final photo in photos) {
        await txn.insert('enrolled_person_photos', {
          'enrolledPersonId': personId,
          'imageData': await photo.readAsBytes(),
        });
      }

      return personId;
    });
  }

  Future<List<EnrolledPerson>> getEnrolledPersonsForPatient(
    int patientId,
  ) async {
    final db = await database;
    final results = await db.query(
      'enrolled_persons',
      where: 'patientId = ?',
      whereArgs: [patientId],
    );
    return results.map((row) => EnrolledPerson.fromMap(row)).toList();
  }

  /// Returns enrolled people with their first locally stored enrollment photo.
  Future<List<EnrolledPerson>> getEnrolledPersonsWithPrimaryPhotoForPatient(
    int patientId,
  ) async {
    final db = await database;
    final results = await db.rawQuery(
      '''
      SELECT enrolled_persons.*,
        (
          SELECT imageData
          FROM enrolled_person_photos
          WHERE enrolledPersonId = enrolled_persons.id
          ORDER BY id ASC
          LIMIT 1
        ) AS primaryPhoto
      FROM enrolled_persons
      WHERE patientId = ?
      ORDER BY id ASC
      ''',
      [patientId],
    );
    return results.map((row) => EnrolledPerson.fromMap(row)).toList();
  }

  // ---------------------------------------------------------------------
  // FUTURE: MemoryEvent methods (Memory Timeline feature) — not yet wired
  // up. MemoryTimelineScreen currently uses static dummy data. Uncomment
  // this block, add the matching CREATE TABLE in onCreate/onUpgrade above,
  // and import 'memory_event.dart' once MemoryEvent is broken out of
  // memory_timeline_screen.dart, to go live.
  // ---------------------------------------------------------------------
  //
  // Future<int> insertMemoryEvent(MemoryEvent event) async {
  //   final db = await database;
  //   return db.insert('memory_events', event.toMap());
  // }
  //
  // Future<List<MemoryEvent>> getMemoryEventsForPatient(int patientId) async {
  //   final db = await database;
  //   final results = await db.query(
  //     'memory_events',
  //     where: 'patientId = ?',
  //     whereArgs: [patientId],
  //     orderBy: 'timestamp DESC',
  //   );
  //   return results.map((row) => MemoryEvent.fromMap(row)).toList();
  // }
  //
  // Future<List<MemoryEvent>> getMemoryEventsForPatientInRange(
  //   int patientId,
  //   DateTime start,
  //   DateTime end,
  // ) async {
  //   final db = await database;
  //   final results = await db.query(
  //     'memory_events',
  //     where: 'patientId = ? AND timestamp BETWEEN ? AND ?',
  //     whereArgs: [patientId, start.toIso8601String(), end.toIso8601String()],
  //     orderBy: 'timestamp DESC',
  //   );
  //   return results.map((row) => MemoryEvent.fromMap(row)).toList();
  // }
  //
  // Future<int> deleteMemoryEvent(int id) async {
  //   final db = await database;
  //   return db.delete('memory_events', where: 'id = ?', whereArgs: [id]);
  // }
}