class Caregiver {
  final int? id;
  final String name;
  final String email;
  final String phone;
  final String passwordHash;
  final String caregiverType;
  final String? familyRelation;
  final String? profession;

  Caregiver({
    this.id,
    required this.name,
    required this.email,
    required this.phone,
    required this.passwordHash,
    required this.caregiverType,
    this.familyRelation,
    this.profession,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'email': email,
      'phone': phone,
      'passwordHash': passwordHash,
      'caregiverType': caregiverType,
      'familyRelation': familyRelation,
      'profession': profession,
    };
  }

  factory Caregiver.fromMap(Map<String, dynamic> map) {
    return Caregiver(
      id: map['id'] as int?,
      name: map['name'] as String,
      email: map['email'] as String,
      phone: map['phone'] as String,
      passwordHash: map['passwordHash'] as String,
      caregiverType: map['caregiverType'] as String,
      familyRelation: map['familyRelation'] as String?,
      profession: map['profession'] as String?,
    );
  }
}