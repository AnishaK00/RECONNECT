import 'dart:typed_data';

class EnrolledPerson {
  final int? id;
  final int patientId;
  final String name;
  final String relation;
  final int photoCount;
  final Uint8List? primaryPhoto;
  final String? photoUrl;

  EnrolledPerson({
    this.id,
    required this.patientId,
    required this.name,
    required this.relation,
    required this.photoCount,
    this.primaryPhoto,
    this.photoUrl,
  });

  Map<String, dynamic> toMap() => {
    'id': id,
    'patientId': patientId,
    'name': name,
    'relation': relation,
    'photoCount': photoCount,
  };

  factory EnrolledPerson.fromMap(Map<String, dynamic> map) => EnrolledPerson(
    id: map['id'] as int?,
    patientId: map['patientId'] as int,
    name: map['name'] as String,
    relation: map['relation'] as String,
    photoCount: map['photoCount'] as int,
    primaryPhoto: map['primaryPhoto'] as Uint8List?,
    photoUrl: map['photo_url'] as String?,
  );
}
