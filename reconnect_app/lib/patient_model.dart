class Patient {
  final int? id;
  final String fullName;
  final String patientId;
  final String username;
  final String passwordHash;
  final String dob;
  final String phone;
  final String email;
  final String address;
  final String caregiverName;
  final String caregiverPhone;
  final String medicalInformation;
  final int caregiverId;

  Patient({
    this.id,
    required this.fullName,
    required this.patientId,
    required this.username,
    required this.passwordHash,
    required this.dob,
    required this.phone,
    required this.email,
    required this.address,
    required this.caregiverName,
    required this.caregiverPhone,
    required this.medicalInformation,
    required this.caregiverId,
  });

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'fullName': fullName,
      'patientId': patientId,
      'username': username,
      'passwordHash': passwordHash,
      'dob': dob,
      'phone': phone,
      'email': email,
      'address': address,
      'caregiverName': caregiverName,
      'caregiverPhone': caregiverPhone,
      'medicalInformation': medicalInformation,
      'caregiverId': caregiverId,
    };
  }

  factory Patient.fromMap(Map<String, dynamic> map) {
    return Patient(
      id: map['id'] as int?,
      fullName: map['fullName'] as String,
      patientId: map['patientId'] as String,
      username: map['username'] as String,
      passwordHash: map['passwordHash'] as String,
      dob: map['dob'] as String,
      phone: map['phone'] as String,
      email: map['email'] as String? ?? '',
      address: map['address'] as String,
      caregiverName: map['caregiverName'] as String,
      caregiverPhone: map['caregiverPhone'] as String,
      medicalInformation: map['medicalInformation'] as String,
      caregiverId: map['caregiverId'] as int,
    );
  }
}