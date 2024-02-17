namespace Drumwise.Application.Common.Interfaces;

public interface IMailSender
{
    Task SendMailAsync(string email, string subject, string body, string bodyType);
}