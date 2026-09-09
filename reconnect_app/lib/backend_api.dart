import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

import 'app_config.dart';

class BackendApi {
  static const _storage = FlutterSecureStorage();
  static const _tokenKey = 'reconnect_auth_token';

  static Future<String?> get token => _storage.read(key: _tokenKey);

  static Future<void> _saveToken(String value) =>
      _storage.write(key: _tokenKey, value: value);

  static Future<Map<String, dynamic>> _json(
    String method,
    String path, {
    Map<String, dynamic>? body,
    bool authenticated = false,
  }) async {
    final headers = {'Content-Type': 'application/json'};
    if (authenticated) {
      final value = await token;
      if (value == null) throw Exception('Please sign in again.');
      headers['Authorization'] = 'Bearer $value';
    }
    final uri = Uri.parse('${AppConfig.httpBaseUrl}$path');
    final response = method == 'GET'
        ? await http.get(uri, headers: headers)
        : await http.post(uri, headers: headers, body: jsonEncode(body));
    final decoded = response.body.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(
        decoded['detail'] ?? 'Request failed (${response.statusCode})',
      );
    }
    return decoded;
  }

  static Future<void> registerCaregiver(Map<String, dynamic> data) async {
    await _json('POST', '/auth/caregivers/register', body: data);
  }

  static Future<void> registerPatient(Map<String, dynamic> data) async {
    await _json('POST', '/auth/patients/register', body: data);
  }

  static Future<Map<String, dynamic>> login({
    required String actorType,
    required String identifier,
    required String password,
  }) async {
    final result = await _json(
      'POST',
      '/auth/login',
      body: {
        'actor_type': actorType,
        'identifier': identifier,
        'password': password,
      },
    );
    await _saveToken(result['token'] as String);
    return result;
  }

  static Future<List<Map<String, dynamic>>> caregiverPatients() async {
    final result = await _json('GET', '/auth/patients', authenticated: true);
    return List<Map<String, dynamic>>.from(result['patients'] as List);
  }

  static Future<List<Map<String, dynamic>>> enrolledPeople(
    int patientId,
  ) async {
    final result = await _json(
      'GET',
      '/enroll/patients/$patientId/people',
      authenticated: true,
    );
    return List<Map<String, dynamic>>.from(result['people'] as List);
  }

  static Future<List<Map<String, dynamic>>> recentSignificantEvents({int limit = 3}) async {
    final result = await _json(
      'GET',
      '/memories/recent?limit=$limit&category=significant_event',
      authenticated: true,
    );
    return List<Map<String, dynamic>>.from(result['memories'] as List);
  }

  static Future<Map<String, dynamic>?> recentMemory() async {
    final result = await _json(
      'GET',
      '/memories/recent?limit=1',
      authenticated: true,
    );
    final memories = List<Map<String, dynamic>>.from(result['memories'] as List);
    return memories.isEmpty ? null : memories.first;
  }

  static Future<List<Map<String, dynamic>>> allMemories() async {
    final result = await _json('GET', '/memories', authenticated: true);
    return List<Map<String, dynamic>>.from(result['memories'] as List);
  }
  

  static Future<Map<String, dynamic>> enrollPerson({
    required int patientId,
    required String name,
    required String relation,
    required String description,
    required List<File> photos,
    required List<File> voiceSamples,
    int? pendingUnknownId,
  }) async {
    final authToken = await token;
    if (authToken == null) throw Exception('Please sign in again.');
    final request =
        http.MultipartRequest(
            'POST',
            Uri.parse('${AppConfig.httpBaseUrl}/enroll/person'),
          )
          ..headers['Authorization'] = 'Bearer $authToken'
          ..fields['patient_id'] = '$patientId'
          ..fields['name'] = name
          ..fields['relation'] = relation;
    if (pendingUnknownId != null) {
      request.fields['pending_unknown_id'] = '$pendingUnknownId';
      }
    for (final photo in photos) {
      request.files.add(
        await http.MultipartFile.fromPath('photos', photo.path),
      );
    }
    for (final sample in voiceSamples) {
      request.files.add(
        await http.MultipartFile.fromPath('voice_samples', sample.path),
      );
    }
    final response = await http.Response.fromStream(await request.send());
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(payload['detail'] ?? 'Enrollment failed');
    }
    return payload;
  }

  static Future<List<Map<String, dynamic>>> pendingUnknowns(
    int patientId,
  ) async {
    final result = await _json(
      'GET',
      '/who-is-this/patients/$patientId/pending',
      authenticated: true,
    );
    return List<Map<String, dynamic>>.from(result['pending'] as List);
  }

  static Future<File> downloadImage(String relativeUrl) async {
    final authToken = await token;
    final response = await http.get(
      Uri.parse('${AppConfig.httpBaseUrl}$relativeUrl'),
      headers: {'Authorization': 'Bearer $authToken'},
    );
    if (response.statusCode != 200){
      throw Exception('Could not download the detected face.');}
    final directory = await getTemporaryDirectory();
    final file = File(
      '${directory.path}/pending_${DateTime.now().microsecondsSinceEpoch}.jpg',
    );
    await file.writeAsBytes(response.bodyBytes);
    return file;
  }

  static Future<Map<String, dynamic>> submitWhoIsThisVideo({
    required int patientId,
    required File video,
  }) async {
    final authToken = await token;
    if (authToken == null) throw Exception('Please sign in again.');
    final request =
        http.MultipartRequest(
            'POST',
            Uri.parse('${AppConfig.httpBaseUrl}/who-is-this/capture'),
          )
          ..headers['Authorization'] = 'Bearer $authToken'
          ..fields['patient_id'] = '$patientId';
    request.files.add(await http.MultipartFile.fromPath('video', video.path));
    final response = await http.Response.fromStream(await request.send());
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(payload['detail'] ?? 'Could not recognize the person.');
    }
    return payload;
  }

  static Future<Map<String, dynamic>> submitUnknownVoiceVideo({
    required int patientId,
    required int unenrolledId,
    required File video,
  }) async {
    final authToken = await token;
    if (authToken == null) throw Exception('Please sign in again.');
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('${AppConfig.httpBaseUrl}/unknown-voice/capture'),
    )
      ..headers['Authorization'] = 'Bearer $authToken'
      ..fields['patient_id'] = '$patientId'
      ..fields['unenrolled_id'] = '$unenrolledId';
    request.files.add(await http.MultipartFile.fromPath('video', video.path));
    final response = await http.Response.fromStream(await request.send());
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(payload['detail'] ?? 'Could not process the unknown voice capture.');
    }
    return payload;
  }
}
