#ifndef __LOG_H__
#define __LOG_H__

#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WIN32
#ifdef __WINSOCK2__
#include <winsock2.h>
#else
#include <winsock.h>
#endif
#include <cassert>
#include <io.h>
#else
#include <unistd.h>
#include <assert.h>
#endif


#include "KernelObject.h"
#include "cttypes.h"

#define MAX_FILE_LINES	(200000)
#define MAX_FILE_SIZE	(20*1024*1024) //20M

#define LOG_TO_NULL		0
#define LOG_TO_FILE		1
#define LOG_TO_SCREEN	2
#define LOG_TO_SCNFILE	3
#define	LOG_TO_UNICODEFILE	4

#ifndef WIN32
#define MAX_PATH			256
#endif
#define MAX_VERSION_LEN		256


//////////////////////////////////////////////////////////////////////////
//��־�¼�����ӿ�
class ILogEventHandler
{
public:
	//////////////////////////////////////////////////////////////////////////
	//��־�����¼�
	//@filename:�ļ���
	//@appName:Ӧ����
	//@moduleName:ģ����	
	//@beginTime:��־��ʼʱ��
	//@endTime:��־����ʱ��
	//@lines:��־����
	virtual	void OnLogBackup(const CTstring& filename, const CTstring& appName, const CTstring& moduleName,
		const CTtime& beginTime, const CTtime& endTime, const CTuint lines)
	{
	}
};

typedef enum 
{
	LogLevel_Closed = 0,
	LogLevel_Fatal = 1,
	LogLevel_Error = 2,
	LogLevel_Warn = 3,
	LogLevel_Info = 4,
	LogLevel_Debug = 5,
}CTLogLevel;

typedef enum
{
	LogBackupT_None = 0, 
	LogBackupT_Quarter = 1,
	LogBackupT_Half = 2,
	LogBackupT_Hour = 3,	
	LogBackupT_Day = 4,
}CTLogBackupType;

class TLog
{
public:
	TLog(const char *szVersion);
	TLog(const char *module, const char *version, int multiuser = 1, const char *logPath = NULL);
	TLog(const char *appname, const char *module,const char *version, const char *logPath = NULL);

	virtual ~TLog();
	//����־�ļ�
	FILE *open(int mode = LOG_TO_FILE,int printVersion = 1);
	//�ر���־�ļ�
	void close();

	//������־�ļ�������Ӧ������ģ�������
	CTbool SetFileName(const char *appname, const char *module);

	//����Ӧ��ģ��İ汾��
	void SetVersion(const char *szVersion);

	//������־��·��
	CTbool SetFilePath(const char *szFilePath);

	//������־�������ʽ����nul,FILE��ȱʡ��,SCREEN or both
	void setMode(int mode);

#ifdef __LOG4J__
	//��log4j�ı���������ʽ
	void useLog4jBackupStyle();
	static void setMaxBackupFile(CTuint uMaxBackupFile);
#endif

	//�����Ƿ�����ˢ����־��־,0=������ˢ�£�1=����ˢ�£�ȱʡ��
	void setFlushFlag(int flag){ m_nFlushFlag = flag;}

	//�����Ƿ��ӡ����,0=������ˢ�£�1=����ˢ�£�ȱʡ��
	void setWithMillisecond(bool flag){ m_hasMillilSecond = flag;}

	//������־�ļ��������
	void setMaxLines(int nMaxLines);
	
	//������־�ļ����ݵ�ʱ��������
	void setBackupTimeIntervalType(int backupType);
	//�õ���־�������ʽ
	int getMode() const { return m_nTraceMode; }

	//��ͨ��־����ͷ��ӡʱ��
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void print(CTLogLevel level, const char *format,...);

	//��ͨ��־����ͷ��ӡʱ���ID
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void print(CTLogLevel level, int ID, const char *format,...);

	//��ͨ��־����ͷ��ӡʱ�䡢ID��qualifier
	//format: MMDD HH:MI:SS <qualifier>[ID]
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void print(CTLogLevel level, CTuint qualifier,CTuint ID, const char *format,...);

	//��ͨ��־����ͷ����ӡʱ��
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void printnt(CTLogLevel level, const char *format,...);

#ifdef __LOG4J__
	//��ͨ��־����ͷ����ӡʱ�����־����
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)
	void printntl(const char *format,...);
#endif

	//��������־����ͷ��ӡʱ��
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void printb(CTLogLevel level, const char *title, const unsigned char *buf,int len);

	//��������־����ͷ��ӡʱ��
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void printb(CTLogLevel level, unsigned int ID, const char *title, const unsigned char *buf,int len);

	//��������־����ͷ����ӡʱ��
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void printnb(CTLogLevel level, const unsigned char *buf,int len);

	//���Դ��ݱ䳤����ָ����д�ӡ,��ͷ��ӡʱ���ID
	void vprint(CTLogLevel level, int ID, const char *format, va_list argptr);

	//format:<guid>[ID]content...
	void vprint(CTLogLevel level, CTuint guid, CTuint ID, const char *format, va_list argptr);

	//��ӡ������־����ͷ��ӡʱ��
	void perror(const char *format,...);

	//ˢ����־
	void flush();

	void SetTraceLevelThreshold(int level);	//���ڶ�̬������־����
	int GetTraceLevelThreshold(){return m_iTraceLevelThreshold;}

	//��ӡ���ⳤ�ȵ����� add by oroming in 2006.07.21
	void printBuffer(CTLogLevel level, const char* pTitle,const char* pBuffer,size_t nLen,CTuint id=0);
	void printBuffer(CTLogLevel level, const char* pBuffer,size_t nLen);
	//////////////////////////////////////////////////////////////////////////
	//ע����־�¼�������
	//hr	2007.02.26
	void registerEventHandler(ILogEventHandler* pHandler);

	void WriteUnicodeFlag();

	//��ͨ��־����ͷ��ӡʱ���ID
	//log4jģʽ�µ���־����ȷ������(��֧��LINUX��WIN���������뼶��Ҫʱ��ȥ�㣬������)���Ҽ�������Զ����
	void print(CTLogLevel level, const wchar_t *content);
private:
	//��ֹ�����÷���printnt(0, "%d", 1234);
	void printnt(int , const char *format,...);
	void perror(int , const char *format,...);

	void printb(const unsigned char *buf, int len);

	//����
	void Backup();

	//��ӡ�汾��Ϣ
	void PrintVersion(void);
	CTbool CanPrint(CTLogLevel level);			//���ڿ����Ƿ��ӡ��־
	CTbool IsNeedBackup();

	inline void LOCK()  { if(m_nMultiUser)  m_Lock.Lock(); }
	inline void UNLOCK()  { if(m_nMultiUser)  m_Lock.Unlock(); }

	int GetMillisecond();


private:
	FILE *m_FP;							//�ļ�������
	CTCriticalSection m_Lock;
#ifdef __LOG4J__
	int m_nUseLog4jStyle; //Ĭ�ϲ�����ʯ��_BAK����ģʽ,���õĻ�������*.log.n��ģʽ��
	int m_nLastBakupIndex; //��Χ[1,10] 1.������ʱ�����ΪĿǰĿ¼��������� 2.������ʱ���ѭ������
	CTuint m_nTotalSize;				//�ܴ�С����
	CTuint m_nMaxSize;					//��־�ļ�������С
	static CTuint m_nMaxBackupFile;
#endif
	char m_acFileName[MAX_PATH];		//��־�ļ���
	char m_acBakFileName[MAX_PATH];		//������־��LOG_Bak
	char m_acErrorFileName[MAX_PATH];	//������־�ļ���
	CTuint m_nTotalCount;				//����������
	CTuint m_nMaxLines;					//��־�ļ����������
	int m_nTraceMode;					//��־���λ�ã�nul,�ļ�,��Ļ,both
	int m_nFlushFlag;					//�Ƿ�����ˢ����־
	int m_nMultiUser;					//�Ƿ���û������û�ʹ��ʱ��Ҫ�������ʣ����û�ʹ��ʱ����Ҫ����
	char m_acVersion[MAX_VERSION_LEN];	//�汾��Ϣ
	int m_iTraceLevelThreshold;
	ILogEventHandler*	m_pEventHandler;//��־�¼�������
	time_t				m_backupFileBeginTime;//������־�ļ��Ŀ�ʼʱ��
	time_t				m_backupFileEndTime;//������־�ļ��Ľ���ʱ��
	CTstring			m_appName;//Ӧ����
	CTstring			m_moduleName;//ģ����
	int m_isPrintVersion;  // ��־�ļ������д�ӡ�汾��Ϣ

	bool m_hasMillilSecond;
	int m_bakcupTimeIntervalType;
private:
	TLog(const TLog& rhs);
	TLog& operator=(const TLog& rhs);
}; 
	
class TLogable
{
public:	
	void registerLogger(TLog* pLog) { m_pLog = pLog;}		
protected:
	TLogable():
		m_pLog(NULL)
		{
		}	

protected:		
	TLog* m_pLog;
};
	
#endif //__LOG_H__

