def save_results(iter, loss, acc, prec, rec, f1):
  # record file existence check
  if not os.path.exists(FLAGS.record_fname):
    with open(FLAGS.record_fname, 'w') as fout:
      fout.write('datetime,data_fname,reg_type,wd,iter,loss,acc,prec,rec,f1')
      fout.write('\n')

  with open(FLAGS.record_fname, 'a') as fout:
    fout.write('{},'.format(FLAGS.timestamp))
    fout.write('{},{},{},'.format(FLAGS.data_fname, FLAGS.reg_type, FLAGS.wd))
    fout.write('{},{},{},{},{},{},'.format(iter, loss, acc, prec, rec, f1))
    fout.write('\n')