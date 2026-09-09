class AppConfig {
  // Toggle this manually depending on where you're testing
  // A physical Android phone must contact the Mac over Wi-Fi, not its own
  // loopback address (127.0.0.1).
  static const bool useSimulator = false;

  static String get serverHost => useSimulator ? '127.0.0.1' : '192.168.0.105'; // ← your Mac's real LAN IP

  static String get wsAudioUrl => 'ws://$serverHost:8000/ws/audio';
  // static String get wsEnrollmentUrl => 'ws://$serverHost:8000/ws/enrollment';
  static String get httpBaseUrl => 'http://$serverHost:8000';
}
