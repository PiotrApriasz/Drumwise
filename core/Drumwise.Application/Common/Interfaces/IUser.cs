namespace Drumwise.Application.Common.Interfaces;

public interface IUser
{
    string? Id { get; }
    Task<string?> GetUserRole();
    Task<bool> IsInRoleAsync(string role);
}