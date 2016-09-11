tempfiles = ['../generator/generator.prototxt', '../descriptor/descriptor_merge.prototxt']
with open('merged.prototxt', 'w') as fo:
  fo.write('name: "texture net"\n')
  for tempfile in tempfiles:
    with open(tempfile, 'r') as fi: fo.write(fi.read())
