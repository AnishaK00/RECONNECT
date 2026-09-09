
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'app_config.dart';

class EnrollmentApiService {
  static Future<Map<String, dynamic>> enrollPerson({
    required String name,
    required String relation,
    required List<File> photos,
    required List<File> voiceSamples,
  }) async {
    final uri = Uri.parse('${AppConfig.httpBaseUrl}/enroll/person');
    final request = http.MultipartRequest('POST', uri)
      ..fields['name'] = name
      ..fields['relation'] = relation;

    for (final photo in photos) {
      request.files.add(await http.MultipartFile.fromPath('photos', photo.path));
    }
    for (final sample in voiceSamples) {
      request.files.add(await http.MultipartFile.fromPath('voice_samples', sample.path));
    }

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode != 200) {
      throw Exception('Enrollment failed: ${response.body}');
    }
    return jsonDecode(response.body) as Map<String, dynamic>;
  }
}