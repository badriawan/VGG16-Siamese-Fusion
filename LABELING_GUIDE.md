# Labeling Guide for Temporal Change Detection

## Cara Mengisi label.txt

Setiap sample temporal harus memiliki file `label.txt` yang berisi satu angka:

```bash
# Untuk sample TANPA perubahan korosi
echo "0" > sample_001/label.txt

# Untuk sample DENGAN perubahan korosi  
echo "1" > sample_002/label.txt
```

## Kriteria Labeling

### Label 0 (No Change) ❌
Gunakan label `0` jika:

- **Jumlah pit korosi:** Tidak ada pit baru yang muncul
- **Ukuran pit existing:** Tidak ada perubahan diameter signifikan (< 5% increase)
- **Kedalaman korosi:** LIDAR depth map menunjukkan kedalaman stabil
- **Aktivitas thermal:** Tidak ada hotspot baru atau peningkatan intensitas thermal
- **Kondisi permukaan:** Tekstur dan morfologi permukaan relatif sama

### Label 1 (Change Detected) ✅
Gunakan label `1` jika:

- **Pit korosi baru:** Muncul pit baru yang tidak ada di "before" image
- **Pertumbuhan pit:** Pit existing bertambah besar (> 5% diameter increase)
- **Perubahan kedalaman:** LIDAR mendeteksi pendalaman korosi (depth value changes)
- **Aktivitas thermal baru:** Hotspot thermal baru atau intensitas bertambah
- **Perubahan morfologi:** Ada perubahan tekstur/bentuk yang signifikan

## Panduan Visual

### RGB Analysis:
- Bandingkan warna dan tekstur permukaan
- Perhatikan area gelap baru (possible pits)
- Cek perubahan reflektivitas permukaan

### Thermal Analysis:
- Identifikasi hotspot baru (warna merah/kuning pada colormap)
- Bandingkan intensitas thermal existing spots
- Perhatikan pola distribusi panas

### LIDAR Analysis:
- Compare depth maps between before/after
- Look for new depression areas (darker in depth map)
- Check for depth value changes in existing pits

## Quality Control Checklist

Sebelum memberikan label, pastikan:

- [ ] **Alignment Check**: Before/after images properly aligned
- [ ] **All Modalities**: RGB, Thermal, dan LIDAR tersedia untuk both timepoints
- [ ] **Clear Visibility**: Area korosi terlihat jelas di semua modalities
- [ ] **Consistent Lighting**: Kondisi pencahayaan relatif sama
- [ ] **Time Gap**: Ada interval waktu yang cukup untuk deteksi perubahan (min. 1 minggu)

## Contoh Labeling

```
temporal_dataset/
├── sample_001/          # No significant change
│   ├── before/
│   ├── after/
│   └── label.txt        → "0"
├── sample_002/          # New pit appeared  
│   ├── before/
│   ├── after/
│   └── label.txt        → "1"
├── sample_003/          # Existing pit grew larger
│   ├── before/
│   ├── after/
│   └── label.txt        → "1"
```

## Tips untuk Konsistensi

1. **Double Check**: Selalu review decision dengan membandingkan semua 3 modalities
2. **Threshold**: Gunakan 5% diameter change sebagai minimum threshold
3. **Documentation**: Catat reasoning untuk ambiguous cases
4. **Inter-rater**: Jika memungkinkan, minta second opinion untuk borderline cases
5. **Systematic**: Process samples dalam urutan yang sistematis

## Common Mistakes

❌ **Jangan:**
- Memberikan label based on single modality saja
- Mengabaikan small but significant changes  
- Labeling noise/artifacts sebagai actual corrosion change
- Inconsistent threshold application

✅ **Lakukan:**
- Analyze all three modalities together
- Use consistent criteria across all samples
- Document uncertain cases
- Cross-validate with domain expert if available

---

**Remember**: Kualitas labeling menentukan performance sistem deteksi. Take time untuk labeling yang accurate dan consistent!