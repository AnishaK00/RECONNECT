class EmergencyContact {
  final int? id;
  final int patientId;
  final String name;
  final String phone;
  final String relationship;

  EmergencyContact({
    this.id,
    required this.patientId,
    required this.name,
    required this.phone,
    required this.relationship,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'patientId': patientId,
      'name': name,
      'phone': phone,
      'relationship': relationship,
    };
  }

  factory EmergencyContact.fromMap(Map<String, dynamic> map) {
    return EmergencyContact(
      id: map['id'] as int?,
      patientId: map['patientId'] as int,
      name: map['name'] as String,
      phone: map['phone'] as String,
      relationship: map['relationship'] as String,
    );
  }
}